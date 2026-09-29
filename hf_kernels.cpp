// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "hf_kernels.h"

#include "slang-cpp-prelude.h"

// Each emit puts main_0 / GlobalParams_0 at file scope under extern "C";
// emptying the EXTERN_C macros lets every emit live in its own namespace in
// one TU (guest/drape/vec_cpu.cpp).
#undef SLANG_PRELUDE_EXTERN_C
#undef SLANG_PRELUDE_EXTERN_C_START
#undef SLANG_PRELUDE_EXTERN_C_END
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END

namespace hk_blend {
#include "../../kernels/headfit/cpp/hf_blend_emit.cpp"
}
namespace hk_blend_bw {
#include "../../kernels/headfit/cpp/hf_blend_backward_emit.cpp"
}
namespace hk_sim {
#include "../../kernels/headfit/cpp/hf_similarity_emit.cpp"
}
namespace hk_sim_bv {
#include "../../kernels/headfit/cpp/hf_similarity_backward_v_emit.cpp"
}
namespace hk_sim_bp {
#include "../../kernels/headfit/cpp/hf_similarity_backward_prm_emit.cpp"
}
namespace hk_surf {
#include "../../kernels/headfit/cpp/hf_surface_residual_emit.cpp"
}
namespace hk_mark {
#include "../../kernels/headfit/cpp/hf_landmark_residual_emit.cpp"
}
namespace hk_prior {
#include "../../kernels/headfit/cpp/hf_prior_emit.cpp"
}
namespace hk_sum {
#include "../../kernels/headfit/cpp/hf_energy_sum_emit.cpp"
}
namespace hk_res {
#include "../../kernels/headfit/cpp/hf_resample_emit.cpp"
}

namespace {

using KernelFn = void (*)(ComputeVaryingInput *, void *, void *);

void dispatch(KernelFn fn, void *gp, uint32_t threads, uint32_t group, uint32_t g0 = 0, uint32_t g1 = hfk::kAll) {
	const uint32_t groups = (threads + group - 1) / group;
	g1 = g1 < groups ? g1 : groups;
	if (threads == 0 || g0 >= g1) {
		return;
	}
	ComputeVaryingInput vi{};
	vi.startGroupID = uint3(g0, 0u, 0u);
	vi.endGroupID = uint3(g1, 1u, 1u);
	fn(&vi, nullptr, gp);
}

// The emits' buffer views take a pointer and a count; the kernels only read
// what the params say, so the count is informational.
template <class B, class T>
void bind(B &b, const T *p, size_t n) {
	b.data = reinterpret_cast<decltype(b.data)>(const_cast<T *>(p));
	b.count = n;
}

} // namespace

namespace hfk {

void blend(uint32_t P, uint32_t NB, const float *base, const float *blend, const float *coeffs, float *v, uint32_t g0,
		uint32_t g1) {
	hk_blend::HfBlendParams_0 pr{ P, NB };
	hk_blend::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.base_0, base, size_t(P) * 3);
	bind(gp.blend_0, blend, size_t(NB) * P * 3);
	bind(gp.coeffs_0, coeffs, NB);
	bind(gp.v_0, v, size_t(P) * 3);
	dispatch(&hk_blend::main_0, &gp, P, 64, g0, g1);
}

void blend_backward(uint32_t P, uint32_t NB, const float *blend, const float *dv, float *dcoeffs, uint32_t g0,
		uint32_t g1) {
	hk_blend_bw::HfBlendParams_0 pr{ P, NB };
	hk_blend_bw::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.blend_0, blend, size_t(NB) * P * 3);
	bind(gp.dv_0, dv, size_t(P) * 3);
	bind(gp.dcoeffs_0, dcoeffs, NB);
	dispatch(&hk_blend_bw::main_0, &gp, NB, 64, g0, g1);
}

void similarity(uint32_t P, const float *prm, const float *v, float *pos, uint32_t g0, uint32_t g1) {
	hk_sim::HfSimParams_0 pr{ P };
	hk_sim::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.prm_0, prm, 10);
	bind(gp.v_0, v, size_t(P) * 3);
	bind(gp.pos_0, pos, size_t(P) * 3);
	dispatch(&hk_sim::main_0, &gp, P, 64, g0, g1);
}

void similarity_backward_v(uint32_t P, const float *prm, const float *dpos, float *dv, uint32_t g0, uint32_t g1) {
	hk_sim_bv::HfSimParams_0 pr{ P };
	hk_sim_bv::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.prm_0, prm, 10);
	bind(gp.dpos_0, dpos, size_t(P) * 3);
	bind(gp.dv_0, dv, size_t(P) * 3);
	dispatch(&hk_sim_bv::main_0, &gp, P, 64, g0, g1);
}

void similarity_backward_prm(uint32_t P, const float *prm, const float *v, const float *dpos, float *dprm) {
	hk_sim_bp::HfSimParams_0 pr{ P };
	hk_sim_bp::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.prm_0, prm, 10);
	bind(gp.v_0, v, size_t(P) * 3);
	bind(gp.dpos_0, dpos, size_t(P) * 3);
	bind(gp.dprm_0, dprm, 10);
	dispatch(&hk_sim_bp::main_0, &gp, 1, 1);
}

void surface_residual(uint32_t P, const float *pos, const float *q, const float *n, const float *w, float *e,
		float *dpos, uint32_t g0, uint32_t g1) {
	hk_surf::HfSurfParams_0 pr{ P };
	hk_surf::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.pos_0, pos, size_t(P) * 3);
	bind(gp.q_0, q, size_t(P) * 3);
	bind(gp.n_0, n, size_t(P) * 3);
	bind(gp.w_0, w, size_t(P) * 2);
	bind(gp.e_0, e, P);
	bind(gp.dpos_0, dpos, size_t(P) * 3);
	dispatch(&hk_surf::main_0, &gp, P, 64, g0, g1);
}

void landmark_residual(uint32_t K, const float *pos, const uint32_t *mid, const float *tgt, const float *lw,
		float *e, float *dpos) {
	hk_mark::HfMarkParams_0 pr{ K };
	hk_mark::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.pos_0, pos, 0);
	bind(gp.mid_0, mid, K);
	bind(gp.tgt_0, tgt, size_t(K) * 3);
	bind(gp.lw_0, lw, K);
	bind(gp.e_0, e, K);
	bind(gp.dpos_0, dpos, 0);
	dispatch(&hk_mark::main_0, &gp, K, 64);
}

void prior(uint32_t N, const float *x, const float *x0, const float *lam, float *e, float *g) {
	hk_prior::HfPriorParams_0 pr{ N };
	hk_prior::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.x_0, x, N);
	bind(gp.x0_0, x0, N);
	bind(gp.lam_0, lam, N);
	bind(gp.e_0, e, N);
	bind(gp.g_0, g, N);
	dispatch(&hk_prior::main_0, &gp, N, 64);
}

double energy_sum(uint32_t N, const float *e) {
	float out[2] = { 0.0f, 0.0f };
	hk_sum::HfSumParams_0 pr{ N };
	hk_sum::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.e_0, e, N);
	bind(gp.out_0, out, 2);
	dispatch(&hk_sum::main_0, &gp, 1, 1);
	return double(out[0]) + double(out[1]);
}

void resample(uint32_t N, const float *prm, const uint32_t *tri, const uint32_t *hit, const float *bary,
		const float *dm, float *out, uint32_t g0, uint32_t g1) {
	hk_res::HfResampleParams_0 pr{ N };
	hk_res::GlobalParams_0 gp{};
	gp.params_0 = &pr;
	bind(gp.prm_0, prm, 10);
	bind(gp.tri_0, tri, 0);
	bind(gp.hit_0, hit, N);
	bind(gp.bary_0, bary, size_t(N) * 3);
	bind(gp.dm_0, dm, 0);
	bind(gp.out_0, out, size_t(N) * 3);
	dispatch(&hk_res::main_0, &gp, N, 64, g0, g1);
}

} // namespace hfk
