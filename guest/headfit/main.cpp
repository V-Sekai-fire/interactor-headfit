// headfit.elf -- the ANNY head fit and the unified-expression transfer (RFD
// 2277 Phase A3, RFD 2275's method) as a sandbox guest: hf_core behind the
// Godot-shaped API, so the same ELF runs under Godot and under the Unity
// sandbox. This TU is the only one that sees api.hpp; hf_core and below are
// std types and link into tests/headfit's native runner too.
//
// Handles are guest ints with one destroy (hf_destroy). Resumable and
// gassable (the operator's rule): hf_data_put only queues; hf_model_load,
// hf_solve and hf_transfer create jobs; hf_tick(handle, budget) advances one
// by at most `budget` units (handle 0: the upload's parser), and the job
// stays in the guest between calls. A fit's tick ends at each accepted
// iterate (AGENTS.md rule 4). Every ADD_API_FUNCTION has a no-argument
// wrapper in project/main.gd (rule 8). No GPU: the CPU path of every Lean
// kernel (kernels/headfit, slangc -target cpp).
// SPDX-License-Identifier: Apache-2.0 OR MIT

#include <api.hpp>

#include <string>
#include <vector>

#include "hf_core.h"

static Variant text(const std::string &s) {
	return Variant(String(s));
}

static Variant hf_data_put(String key, PackedArray<uint8_t> bytes) {
	const std::vector<uint8_t> v = bytes.fetch();
	hf::data_put(key.utf8(), std::string(v.begin(), v.end()));
	return Variant(int64_t(hf::data_queued()));
}

static Variant hf_data_clear() {
	hf::data_clear();
	return Variant(int64_t(0));
}

static Variant hf_data_report() {
	return text(hf::data_report());
}

static Variant hf_mesh_create(PackedArray<float> xyz, PackedArray<int32_t> tri) {
	const std::vector<float> p = xyz.fetch();
	const std::vector<int32_t> t = tri.fetch();
	return Variant(int64_t(hf::mesh_create(p.data(), int(p.size() / 3), t.data(), int(t.size() / 3))));
}

static Variant hf_mesh_add_shape(int mesh, String name, PackedArray<float> deltas) {
	const std::vector<float> d = deltas.fetch();
	return Variant(int64_t(hf::mesh_add_shape(mesh, name.utf8(), d.data(), int(d.size()))));
}

static Variant hf_model_load(int region) {
	return Variant(int64_t(hf::model_load(region)));
}

static Variant hf_marks_create(PackedArray<int32_t> model_ids, PackedArray<int32_t> avatar_ids) {
	const std::vector<int32_t> m = model_ids.fetch(), a = avatar_ids.fetch();
	if (m.size() != a.size()) {
		return Variant(int64_t(-1));
	}
	return Variant(int64_t(hf::marks_create(m.data(), a.data(), int(m.size()))));
}

static Variant hf_solve(int model, int mesh, int marks, PackedArray<float> weights) {
	const std::vector<float> w = weights.fetch();
	return Variant(int64_t(hf::solve(model, mesh, marks, w.data(), int(w.size()))));
}

static Variant hf_tick(int handle, int64_t budget) {
	return text(hf::tick(handle, budget));
}

static Variant hf_done(int handle) {
	return Variant(hf::done(handle));
}

static Variant hf_fit_residual_mm(int fit, int region) {
	return Variant(double(hf::fit_residual_mm(fit, region)));
}

static Variant hf_transfer(int fit, String shape) {
	return Variant(int64_t(hf::transfer(fit, shape.utf8())));
}

static Variant hf_transfer_result(int job) {
	std::vector<float> d;
	float err = 0.0f;
	std::string info;
	Dictionary r = Dictionary::Create();
	if (!hf::transfer_result(job, d, err, info)) {
		r["error"] = text(hf::last_error());
		return Variant(r);
	}
	r["deltas"] = Variant(PackedArray<float>(d));
	r["error_mm"] = Variant(double(err));
	r["info"] = text(info);
	return Variant(r);
}

static Variant hf_destroy(int handle) {
	return Variant(hf::destroy(handle));
}

static Variant hf_live() {
	return Variant(int64_t(hf::live_handles()));
}

static Variant hf_error() {
	return text(hf::last_error());
}

static Variant hf_model_report(int model) {
	return text(hf::model_report(model));
}

static Variant hf_fit_report(int fit) {
	return text(hf::fit_report(fit));
}

static Variant hf_fit_x(int fit) {
	std::vector<float> x;
	hf::fit_x(fit, x);
	return Variant(PackedArray<float>(x));
}

static Variant hf_map_audit(String required, int mesh) {
	return text(hf::map_audit(required.utf8(), mesh));
}

int main() {
	ADD_API_FUNCTION(hf_data_put, "int", "String key, PackedByteArray bytes",
			"Queue one ANNY source (base.obj before any target) or the map / dials text; returns the queue length");
	ADD_API_FUNCTION(hf_data_clear, "int", "", "Drop every upload and parsed source");
	ADD_API_FUNCTION(hf_data_report, "String", "", "base.obj's counts and each parsed target's checksums");
	ADD_API_FUNCTION(hf_mesh_create, "int", "PackedFloat32Array xyz, PackedInt32Array triangles",
			"The avatar mesh (metres); a handle");
	ADD_API_FUNCTION(hf_mesh_add_shape, "int", "int mesh, String name, PackedFloat32Array deltas",
			"One of the avatar's own shapes (dense deltas, metres), for artist: sources");
	ADD_API_FUNCTION(hf_model_load, "int", "int region", "A model job over ANNY's head (0) or face (1); tick it");
	ADD_API_FUNCTION(hf_marks_create, "int", "PackedInt32Array model_ids, PackedInt32Array avatar_ids",
			"Landmark pairs: base.obj vertex (0-based) and avatar vertex");
	ADD_API_FUNCTION(hf_solve, "int", "int model, int mesh, int marks, PackedFloat32Array weights",
			"A fit job (weights in hf_core.h's order; empty for the defaults); tick it");
	ADD_API_FUNCTION(hf_tick, "String", "int handle, int budget",
			"Advance a job (0: the upload's parser) by at most budget units");
	ADD_API_FUNCTION(hf_done, "bool", "int handle", "Is the job done");
	ADD_API_FUNCTION(hf_fit_residual_mm, "float", "int fit, int region", "RMS point-to-surface distance of a region, mm");
	ADD_API_FUNCTION(hf_transfer, "int", "int fit, String shape",
			"A transfer job: an ANNY action or a unified name onto the avatar; tick it");
	ADD_API_FUNCTION(hf_transfer_result, "Dictionary", "int job", "{deltas, error_mm, info} or {error}");
	ADD_API_FUNCTION(hf_destroy, "bool", "int handle", "Free any handle");
	ADD_API_FUNCTION(hf_live, "int", "", "Live handles");
	ADD_API_FUNCTION(hf_error, "String", "", "The last error");
	ADD_API_FUNCTION(hf_model_report, "String", "int model", "The model's counts, labels and actions");
	ADD_API_FUNCTION(hf_fit_report, "String", "int fit", "The fit's unknowns, rounds, residuals and landmarks");
	ADD_API_FUNCTION(hf_fit_x, "PackedFloat32Array", "int fit", "The fit's unknowns (float32)");
	ADD_API_FUNCTION(hf_map_audit, "String", "String required, int mesh",
			"unified_expressions.map against the required names, per stage");
	halt();
}
