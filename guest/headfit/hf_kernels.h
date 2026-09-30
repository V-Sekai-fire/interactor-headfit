// hf_kernels -- the head fit's Lean-emitted kernels (kernels/headfit, the
// slangc cpp target) behind plain calls on host vectors: the CPU path,
// one dispatch at a time, the guest/drape/vec_cpu.cpp pattern. No kernel
// arithmetic lives here; each function binds buffers and dispatches.
// SPDX-License-Identifier: Apache-2.0 OR MIT
#pragma once

#include <cstdint>
#include <vector>

namespace hfk {

// Per-thread kernels take a range of 64-thread groups [g0, g1) (clamped): a
// fit's items dispatch one group at a time. Each thread writes only its own
// outputs, so any split computes the same bits.
constexpr uint32_t kAll = 0xFFFFFFFFu;

// v = base + sum_c coeffs[c] blend[c]           (hf_blend)
void blend(uint32_t P, uint32_t NB, const float *base, const float *blend, const float *coeffs, float *v, uint32_t g0 = 0,
		uint32_t g1 = kAll);
// dcoeffs = blend^T dv (df32)                    (hf_blend_backward)
void blend_backward(uint32_t P, uint32_t NB, const float *blend, const float *dv, float *dcoeffs, uint32_t g0 = 0,
		uint32_t g1 = kAll);
// pos = s R v + t, prm = [rot6, t, s]            (hf_similarity)
void similarity(uint32_t P, const float *prm, const float *v, float *pos, uint32_t g0 = 0, uint32_t g1 = kAll);
// dv = s R^T dpos                                (hf_similarity_backward_v)
void similarity_backward_v(uint32_t P, const float *prm, const float *dpos, float *dv, uint32_t g0 = 0, uint32_t g1 = kAll);
// dprm = d(rot6, t, s) (serial, df32)            (hf_similarity_backward_prm)
void similarity_backward_prm(uint32_t P, const float *prm, const float *v, const float *dpos, float *dprm);
// point-to-plane + point-to-point               (hf_surface_residual)
void surface_residual(uint32_t P, const float *pos, const float *q, const float *n, const float *w, float *e,
		float *dpos, uint32_t g0 = 0, uint32_t g1 = kAll);
// landmarks, dpos += ...                         (hf_landmark_residual)
void landmark_residual(uint32_t K, const float *pos, const uint32_t *mid, const float *tgt, const float *lw,
		float *e, float *dpos);
// prior                                          (hf_prior)
void prior(uint32_t N, const float *x, const float *x0, const float *lam, float *e, float *g);
// sum of e as a df32 pair, returned in double    (hf_energy_sum)
double energy_sum(uint32_t N, const float *e);
// out = s R sum_k bary_k dm[tri_k] at each hit   (hf_resample)
void resample(uint32_t N, const float *prm, const uint32_t *tri, const uint32_t *hit, const float *bary,
		const float *dm, float *out, uint32_t g0 = 0, uint32_t g1 = kAll);

} // namespace hfk
