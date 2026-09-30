// hf_core -- the head fit and the unified-expression transfer (RFD 2275's
// method, RFD 2277 Phase A3), shared by headfit.elf (guest/headfit/main.cpp)
// and the native check runner (tests/headfit/). No Eigen, no filesystem, no
// Python: ANNY's CC0 sources arrive as text through data_put, and every
// residual and gradient is a Lean-emitted kernel (kernels/headfit, cpp
// target) called through hf_kernels.h. The solver is dress-on's L-BFGS-B
// (guest/drape/lbfgsb.h) in reverse communication.
//
// Resumable and gassable (the operator's rule for sandbox work): the upload's
// parse, a model's build, a fit and a transfer are jobs. Creating one does
// bounded work; tick(handle, budget) advances it by items until the budget
// of units (about 1024 inner-loop iterations each) is spent, at least one
// item a tick, and the job's state stays in the guest between ticks. Any
// slicing gives the same bits: items never split an arithmetic sequence
// differently by budget. A fit's tick also ends at each accepted iterate
// and round (AGENTS.md rule 4: the host ticks it, one step at a time).
//
// Units: the avatar mesh is in metres (Godot and Unity), ANNY's MPFB2 files
// are in decimetres; inside, both are millimetres, so the loss reads as a
// weighted mean squared distance in mm^2 and the residuals in mm.
//
// Handles are positive ints, one table for every kind, one destroy.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hf {

// Region ids (hf_fit_residual_mm, hf_model_load, the loss weights).
enum Region : int {
	R_ALL = 0, // the fit region: the head, every label but the neck band
	R_FACE = 1, // the facial actions' support, less the three below
	R_MOUTH = 2, // the mouth dials' support
	R_NOSE = 3, // the nose dials' support (down-weighted)
	R_EYES = 4, // the eyelids (masked out of the loss)
	R_CRANIUM = 5, // the rest of the head
	R_NECK = 6, // the neck band below the head: surface for the transfer (neck-platysma), weight 0 in the fit
	R_COUNT = 7,
};
const char *region_name(int r);

// The loss weights hf_solve takes, in this order; a shorter array keeps the
// defaults for the rest.
enum Weight : int {
	W_FACE, W_MOUTH, W_NOSE, W_EYES, W_CRANIUM, // per-region point-to-surface weights
	W_LANDMARK, // the landmark term, spread over the marks
	W_PRIOR_PHEN, W_PRIOR_DIAL, // the prior on the phenotype (to 0.5) and dials (to 0)
	W_POINT, // the point-to-point share next to point-to-plane
	W_ROUNDS, // correspondence rounds
	W_ITERS, // L-BFGS-B iterations per round
	W_MAXDIST, // correspondence cut-off, mm
	W_COUNT,
};
extern const float kWeightDefaults[W_COUNT];

// ANNY's CC0 sources, by their path under ANNY's data/ directory
// ("mpfb2/3dobjs/base.obj", "faceunits01/targets/faceunits/jawOpen.target",
// "mpfb2/targets/macrodetails/height/...target"), as plain text: the host
// inflates the .gz. Also "unified_expressions.map" and "local_dials.txt".
// Queue a file for the loader (parsed by ticks; base.obj before any target).
void data_put(const std::string &key, const std::string &text);
bool data_has(const std::string &key);
size_t data_count();
size_t data_queued();
void data_clear();

const std::string &last_error();
// base.obj's counts and each target file's checksums (the load gate).
std::string data_report();

int mesh_create(const float *xyz, int nv, const int32_t *tri, int nt);
int mesh_add_shape(int mesh, const std::string &name, const float *deltas, int n3);
// A model job: parses the queued uploads, then builds (tick it until DONE).
int model_load(int region);
int marks_create(const int32_t *model_ids, const int32_t *avatar_ids, int n);
// A fit job on a built model (tick it until DONE).
int solve(int model, int mesh, int marks, const float *w, int nw);
// Advance a job (0: the loader) by at most `budget` units. "RUNNING ...",
// "DONE ...", "FAIL ...".
std::string tick(int handle, int64_t budget);
bool done(int handle);
float fit_residual_mm(int fit, int region);
// A transfer job on a finished fit: one of ANNY's 52 actions or a name of
// unified_expressions.map. Its result: 3 floats per avatar vertex, metres,
// in the mesh's frame, and the round-trip error in mm.
int transfer(int fit, const std::string &shape);
bool transfer_result(int job, std::vector<float> &deltas, float &error_mm, std::string &info);
bool destroy(int handle);
int live_handles();
// Reports (one line per fact; the gates parse them).
std::string model_report(int model);
std::string fit_report(int fit);
// The catalog audit against the required names (one per line), per stage;
// with a mesh handle (> 0) its artist: sources are checked against the
// shapes the host added to that mesh.
std::string map_audit(const std::string &required_names, int mesh = 0);
// The fit's unknowns (x, float32) and the posed model (mm), for the oracle.
bool fit_x(int fit, std::vector<float> &x);
// A transfer job of a model-space delta field (P*3, mm) through the fit's
// hits, for the identity gate; its result is transfer_result's.
int transfer_field(int fit, const std::vector<float> &dm);

// The model region at a phenotype (six axes, dials 0) as a mesh in metres in
// base.obj's frame, its triangles and each vertex's base.obj index: the
// transfer identity gate's target. And one of the 52 actions on the region
// (P*3, metres).
bool model_mesh(int model, const double phen[6], std::vector<float> &xyz_m, std::vector<int32_t> &tri,
		std::vector<int32_t> &gid);
bool model_action(int model, const std::string &action, std::vector<float> &d_m);

// Coefficient rule (anny_coeffs.gd), exposed for the parity check: the
// macro rows' names and their coefficients at a phenotype (six axes).
bool macro_coefficients(int model, const double phen[6], std::vector<std::string> &names, std::vector<double> &c);

// Everything a native oracle needs to restate the objective of one
// correspondence round in double (tests/headfit/oracle.cpp).
struct RoundProblem {
	uint32_t P = 0, NB = 0, K = 0, n = 0, L = 0;
	std::vector<float> base, blend; // P*3, NB*P*3 (mm, model frame)
	std::vector<float> q, nrm, w; // P*3, P*3, P*2
	std::vector<uint32_t> mid; // K
	std::vector<float> tgt, lw; // K*3, K
	std::vector<float> x0, lb, ub, prior_x0, prior_lam; // n
	uint32_t macro_rows = 0; // rows [0, macro_rows) are macro, then dial pos/neg pairs
	std::vector<std::vector<int>> row_slots; // per macro row, its slot indices
	std::vector<std::vector<double>> anchors; // per axis
	std::vector<std::vector<int>> axis_slots; // per axis
};
// Round 0's problem as the fit set it up (after the landmark initialisation).
bool round_problem(int fit, int round, RoundProblem &out);
// The fit's x at the end of each round.
bool round_result(int fit, int round, std::vector<float> &x, double &f, int &iters);
// The loss and its gradient at x on the current round's problem, by the
// guest's own pipeline (the oracle's gradient check), and the model's base
// and blend rows (mm, model frame).
bool evaluate_at(int fit, const std::vector<float> &x, double &f, std::vector<float> &g);
bool model_blend(int model, std::vector<float> &base, std::vector<float> &blend);

} // namespace hf
