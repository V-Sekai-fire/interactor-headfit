// The C++ emission of the SOMA-X pose kernels, one translation unit each: each unit includes its
// kernel's emitted kernels/anny/cpp/<k>_emit.cpp (Slang prelude inlined) and exports one C entry
// point. build.sh compiles this file once per kernel with -DSOMA_KERNEL=<n> and keeps only that
// unit's *_run symbol global, so every kernel links into one test.
#include <cstddef>
#include <cstdint>

#define BUF(name, ptr, n) g.name##_0.data = (ptr); g.name##_0.count = (n)
#define GO(n)                                            \
	ComputeVaryingInput vi{};                            \
	vi.startGroupID = uint3(0u, 0u, 0u);                 \
	vi.endGroupID = uint3(((n) + 63u) / 64u, 1u, 1u);    \
	main_0(&vi, nullptr, &g)

#if SOMA_KERNEL == 1
#include "soma_orient_emit.cpp"
extern "C" void soma_orient_run(uint32_t NJ, uint32_t gtj, const float *pose, const float *transl, const float *orient,
		const float *orientPT, const float *localT, const uint32_t *parents, const float *corrBind, float *rabs,
		float *acc, uint32_t *anc, float *feat) {
	SomaOrientParams_0 p{ NJ, gtj };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(pose, (float *)pose, 9 * NJ); BUF(transl, (float *)transl, 3); BUF(orient, (float *)orient, 9 * NJ);
	BUF(orientPT, (float *)orientPT, 9 * NJ); BUF(localT, (float *)localT, 3 * NJ);
	BUF(parents, (uint32_t *)parents, NJ); BUF(corrBind, (float *)corrBind, 9 * NJ);
	BUF(rabs, rabs, 9 * NJ); BUF(acc, acc, 12 * NJ); BUF(anc, anc, NJ); BUF(feat, feat, 6 * NJ);
	GO(NJ);
}
#elif SOMA_KERNEL == 2
#include "soma_jump_emit.cpp"
extern "C" void soma_jump_run(uint32_t N, const float *accIn, const uint32_t *ancIn, float *accOut, uint32_t *ancOut) {
	SomaJumpParams_0 p{ N };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(accIn, (float *)accIn, 12 * N); BUF(ancIn, (uint32_t *)ancIn, N);
	BUF(accOut, accOut, 12 * N); BUF(ancOut, ancOut, N);
	GO(N);
}
#elif SOMA_KERNEL == 3
#include "soma_twist_emit.cpp"
extern "C" void soma_twist_run(uint32_t NJ, uint32_t NS, uint32_t twistForm, const float *rabs, const float *world, const uint32_t *srcAxis,
		const float *srcSign, const uint32_t *segStart, const uint32_t *segEnd, const uint32_t *segParent,
		const uint32_t *segRev, const float *bindRot, const float *alignRot, float *chan) {
	SomaTwistParams_0 p{ NJ, NS, twistForm };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(rabs, (float *)rabs, 9 * NJ); BUF(world, (float *)world, 12 * NJ);
	BUF(srcAxis, (uint32_t *)srcAxis, NJ); BUF(srcSign, (float *)srcSign, NJ);
	BUF(segStart, (uint32_t *)segStart, NS); BUF(segEnd, (uint32_t *)segEnd, NS);
	BUF(segParent, (uint32_t *)segParent, NS); BUF(segRev, (uint32_t *)segRev, NS);
	BUF(bindRot, (float *)bindRot, 9 * NJ); BUF(alignRot, (float *)alignRot, 9 * NS); BUF(chan, chan, 3 * NJ);
	GO(NJ);
}
#elif SOMA_KERNEL == 4
#include "soma_expand_emit.cpp"
extern "C" void soma_expand_run(uint32_t NT, uint32_t NJ, uint32_t NTW, const uint32_t *kind, const uint32_t *refIdx,
		const uint32_t *par, const float *baseRot, const float *localT, const float *twistM, const float *chan,
		const uint32_t *twAxis, const float *twSign, const float *publicWorld, float *acc, uint32_t *anc) {
	SomaExpandParams_0 p{ NT, NJ, NTW };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(kind, (uint32_t *)kind, NT); BUF(refIdx, (uint32_t *)refIdx, NT); BUF(par, (uint32_t *)par, NT);
	BUF(baseRot, (float *)baseRot, 9 * NT); BUF(localT, (float *)localT, 3 * NT);
	BUF(twistM, (float *)twistM, 3 * NTW * NJ); BUF(chan, (float *)chan, 3 * NJ);
	BUF(twAxis, (uint32_t *)twAxis, NTW); BUF(twSign, (float *)twSign, NTW);
	BUF(publicWorld, (float *)publicWorld, 12 * NJ); BUF(acc, acc, 12 * NT); BUF(anc, anc, NT);
	GO(NT);
}
#elif SOMA_KERNEL == 5
#include "soma_skin_emit.cpp"
extern "C" void soma_skin_run(uint32_t NT, const float *world, const float *invBind, float *bone) {
	SomaSkinParams_0 p{ NT };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(world, (float *)world, 12 * NT); BUF(invBind, (float *)invBind, 12 * NT); BUF(bone, bone, 12 * NT);
	GO(NT);
}
#elif SOMA_KERNEL == 6
#include "soma_corr_l1_emit.cpp"
extern "C" void soma_corr_l1_run(uint32_t D, uint32_t K, uint32_t useTanh, const float *feat, const float *w1, float *z) {
	SomaCorrL1Params_0 p{ D, K, useTanh };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(feat, (float *)feat, D); BUF(w1, (float *)w1, size_t(D) * K); BUF(z, z, K);
	GO(K);
}
#elif SOMA_KERNEL == 7
#include "soma_corr_l2_emit.cpp"
extern "C" void soma_corr_l2_run(uint32_t V, uint32_t K, uint32_t E, const float *rest, const uint32_t *rowPtr,
		const uint32_t *col, const float *val, const float *z, float *bind) {
	SomaCorrL2Params_0 p{ V };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(rest, (float *)rest, 3 * V); BUF(rowPtr, (uint32_t *)rowPtr, V + 1); BUF(col, (uint32_t *)col, E);
	BUF(val, (float *)val, 3 * size_t(E)); BUF(z, (float *)z, K); BUF(bind, bind, 3 * V);
	GO(V);
}
#elif SOMA_KERNEL == 8
#include "anny_lbs_emit.cpp"
extern "C" void anny_lbs_run(uint32_t V, uint32_t J, const float *bind, const float *weights, const float *bone, float *verts) {
	AnnyLbsParams_0 p{ V, J };
	GlobalParams_0 g{};
	g.params_0 = &p;
	BUF(bind, (float *)bind, 3 * V); BUF(weights, (float *)weights, size_t(V) * J); BUF(bone, (float *)bone, 12 * J);
	BUF(verts, verts, 3 * V);
	GO(V);
}
#endif
