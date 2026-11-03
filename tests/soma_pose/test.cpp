// SPDX-License-Identifier: MIT
//
// Parity of the SOMA-X pose kernels (kernels/anny soma_* + anny_lbs, their slangc C++ emission) with
// SOMA-X itself, on the fixture tools/soma/export_soma.py writes.
//
//   soma_pose_test <fixture dir>             the C++ emission (CPU)
//   soma_pose_test <fixture dir> --gpu <dir> the GPU run's vertices (gpu_run.gd output), same gates
//
// Per clip, every sampled frame runs the whole per-frame chain (orient, public FK, twist channels,
// target expansion, target FK, skinning matrices, correctives, LBS) from that frame's pose and
// translation, and is compared with SOMA-X's vertices for the same frame:
//
//   parity       the kernels as they are                       must PASS
//   control      correctives off (W2 zero)                     must FAIL
//   control      twist joints skipped (twist matrix zero)      must FAIL
//   control      one frame late (input frame f + 1)            must FAIL
//
// The gate is stated with its reason at VERT_BOUND_M below. Exit 0 only if every parity passes and
// every control fails.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void soma_orient_run(uint32_t, uint32_t, const float *, const float *, const float *, const float *, const float *,
		const uint32_t *, const float *, float *, float *, uint32_t *, float *);
void soma_jump_run(uint32_t, const float *, const uint32_t *, float *, uint32_t *);
void soma_twist_run(uint32_t, uint32_t, uint32_t, const float *, const float *, const uint32_t *, const float *, const uint32_t *,
		const uint32_t *, const uint32_t *, const uint32_t *, const float *, const float *, float *);
void soma_expand_run(uint32_t, uint32_t, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, const float *,
		const float *, const float *, const float *, const uint32_t *, const float *, const float *, float *, uint32_t *);
void soma_skin_run(uint32_t, const float *, const float *, float *);
void soma_corr_l1_run(uint32_t, uint32_t, uint32_t, const float *, const float *, float *);
void soma_corr_l2_run(uint32_t, uint32_t, uint32_t, const float *, const uint32_t *, const uint32_t *, const float *,
		const float *, float *);
void anny_lbs_run(uint32_t, uint32_t, const float *, const float *, const float *, float *);
}

// The vertex gate. SOMA-X runs in float32, and so do the kernels, but they round in a different
// order (torch's blocked matmuls and level-order FK against straight-line sums and pointer jumping).
// Each float32 operation is off by at most half an ulp, 2^-24 relative. The longest chain a vertex
// goes through is about 2^10 rounded operations (the corrective sum alone is up to ~10^3 terms per
// vertex, then FK, skinning and LBS add a few dozen), each at most 2^-24 of the magnitudes involved.
// Those magnitudes are the vertex coordinates themselves (the clips carry the body up to 13 m from
// the origin), so the bound per clip is 2^10 * 2^-24 * M, M = the largest |coordinate| SOMA-X
// outputs over the clip. It is a worst case, not the expected error (rounding errors mostly
// cancel); the measured error is printed beside it.
static double vertBound(const std::vector<float> &v) {
	double M = 0;
	for (float x : v) M = std::max(M, double(std::fabs(x)));
	return 1024.0 * std::ldexp(1.0, -24) * M;
}

static std::string slurp(const std::string &p) {
	std::ifstream f(p);
	if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
	std::stringstream s; s << f.rdbuf(); return s.str();
}
static long metaInt(const std::string &m, const char *key) {
	std::string k = std::string("\"") + key + "\": ";
	size_t i = m.find(k);
	if (i == std::string::npos) { fprintf(stderr, "meta: no %s\n", key); exit(2); }
	return atol(m.c_str() + i + k.size());
}
template <class T> static std::vector<T> load(const std::string &p, size_t n) {
	std::vector<T> v(n);
	FILE *f = fopen(p.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); }
	size_t got = fread(v.data(), sizeof(T), n, f);
	fclose(f);
	if (got != n) { fprintf(stderr, "%s: %zu of %zu\n", p.c_str(), got, n); exit(2); }
	return v;
}

struct Rig {
	uint32_t NJ, NT, NS, NTW, V, D, K, E, gtj, useTanh, roundsPub, roundsTgt;
	std::vector<uint32_t> pubParent, srcAxis, segStart, segEnd, segParent, segRev, twAxis, kind, ref, par, rowPtr, col;
	std::vector<float> pubLocalT, orient, orientPT, corrBind, srcSign, bindRot, alignRot, twistM, twSign, baseRot,
			tgtLocalT, invBind, rest, weights, w1, w2val;
};

static Rig loadRig(const std::string &dir) {
	std::string m = slurp(dir + "/meta.json");
	Rig r;
	r.NJ = metaInt(m, "NJ"); r.NT = metaInt(m, "NT"); r.NS = metaInt(m, "NS"); r.NTW = metaInt(m, "NTW");
	r.V = metaInt(m, "V"); r.D = metaInt(m, "D"); r.K = metaInt(m, "K"); r.E = metaInt(m, "E");
	r.gtj = metaInt(m, "gtj"); r.useTanh = metaInt(m, "use_tanh");
	r.roundsPub = metaInt(m, "rounds_pub"); r.roundsTgt = metaInt(m, "rounds_tgt");
	std::string t = dir + "/tables/";
	r.pubParent = load<uint32_t>(t + "pub_parent.bin", r.NJ);
	r.pubLocalT = load<float>(t + "pub_local_t.bin", 3 * r.NJ);
	r.orient = load<float>(t + "orient.bin", 9 * r.NJ);
	r.orientPT = load<float>(t + "orient_pt.bin", 9 * r.NJ);
	r.corrBind = load<float>(t + "corr_bind.bin", 9 * r.NJ);
	r.srcAxis = load<uint32_t>(t + "src_axis.bin", r.NJ);
	r.srcSign = load<float>(t + "src_sign.bin", r.NJ);
	r.segStart = load<uint32_t>(t + "seg_start.bin", r.NS);
	r.segEnd = load<uint32_t>(t + "seg_end.bin", r.NS);
	r.segParent = load<uint32_t>(t + "seg_parent.bin", r.NS);
	r.segRev = load<uint32_t>(t + "seg_rev.bin", r.NS);
	r.bindRot = load<float>(t + "bind_rot.bin", 9 * r.NJ);
	r.alignRot = load<float>(t + "align_rot.bin", 9 * r.NS);
	r.twistM = load<float>(t + "twist_m.bin", 3 * r.NTW * r.NJ);
	r.twAxis = load<uint32_t>(t + "tw_axis.bin", r.NTW);
	r.twSign = load<float>(t + "tw_sign.bin", r.NTW);
	r.kind = load<uint32_t>(t + "tgt_kind.bin", r.NT);
	r.ref = load<uint32_t>(t + "tgt_ref.bin", r.NT);
	r.par = load<uint32_t>(t + "tgt_par.bin", r.NT);
	r.baseRot = load<float>(t + "tgt_base_rot.bin", 9 * r.NT);
	r.tgtLocalT = load<float>(t + "tgt_local_t.bin", 3 * r.NT);
	r.invBind = load<float>(t + "inv_bind.bin", 12 * r.NT);
	r.rest = load<float>(t + "rest.bin", 3 * r.V);
	r.weights = load<float>(t + "weights.bin", size_t(r.V) * r.NT);
	r.w1 = load<float>(t + "w1.bin", size_t(r.D) * r.K);
	r.rowPtr = load<uint32_t>(t + "w2_rowptr.bin", r.V + 1);
	r.col = load<uint32_t>(t + "w2_col.bin", r.E);
	r.w2val = load<float>(t + "w2_val.bin", 3 * size_t(r.E));
	return r;
}

// Rounds of pointer jumping, ping-ponging between (a0, n0) and (a1, n1); returns the result buffer.
static float *jump(uint32_t N, uint32_t rounds, std::vector<float> &a0, std::vector<uint32_t> &n0,
		std::vector<float> &a1, std::vector<uint32_t> &n1) {
	float *ai = a0.data(), *ao = a1.data();
	uint32_t *ni = n0.data(), *no = n1.data();
	for (uint32_t k = 0; k < rounds; ++k) {
		soma_jump_run(N, ai, ni, ao, no);
		std::swap(ai, ao); std::swap(ni, no);
	}
	return ai;
}

struct Frame {
	std::vector<float> rabs, accA, accB, feat, chan, taccA, taccB, bone, z, bind, verts;
	std::vector<uint32_t> ancA, ancB, tancA, tancB;
	float *pubWorld = nullptr, *world = nullptr;
	explicit Frame(const Rig &r)
		: rabs(9 * r.NJ), accA(12 * r.NJ), accB(12 * r.NJ), feat(6 * r.NJ), chan(3 * r.NJ), taccA(12 * r.NT),
		  taccB(12 * r.NT), bone(12 * r.NT), z(r.K), bind(3 * r.V), verts(3 * r.V), ancA(r.NJ), ancB(r.NJ),
		  tancA(r.NT), tancB(r.NT) {}
};

static void poseFrame(const Rig &r, const float *pose, const float *transl, const float *twistM, const float *w2val,
		Frame &f, uint32_t twistForm = 1) {
	soma_orient_run(r.NJ, r.gtj, pose, transl, r.orient.data(), r.orientPT.data(), r.pubLocalT.data(),
			r.pubParent.data(), r.corrBind.data(), f.rabs.data(), f.accA.data(), f.ancA.data(), f.feat.data());
	f.pubWorld = jump(r.NJ, r.roundsPub, f.accA, f.ancA, f.accB, f.ancB);
	soma_twist_run(r.NJ, r.NS, twistForm, f.rabs.data(), f.pubWorld, r.srcAxis.data(), r.srcSign.data(), r.segStart.data(),
			r.segEnd.data(), r.segParent.data(), r.segRev.data(), r.bindRot.data(), r.alignRot.data(), f.chan.data());
	soma_expand_run(r.NT, r.NJ, r.NTW, r.kind.data(), r.ref.data(), r.par.data(), r.baseRot.data(),
			r.tgtLocalT.data(), twistM, f.chan.data(), r.twAxis.data(), r.twSign.data(), f.pubWorld,
			f.taccA.data(), f.tancA.data());
	f.world = jump(r.NT, r.roundsTgt, f.taccA, f.tancA, f.taccB, f.tancB);
	soma_skin_run(r.NT, f.world, r.invBind.data(), f.bone.data());
	soma_corr_l1_run(r.D, r.K, r.useTanh, f.feat.data(), r.w1.data(), f.z.data());
	soma_corr_l2_run(r.V, r.K, r.E, r.rest.data(), r.rowPtr.data(), r.col.data(), w2val, f.z.data(), f.bind.data());
	anny_lbs_run(r.V, r.NT, f.bind.data(), r.weights.data(), f.bone.data(), f.verts.data());
}

struct Stat {
	double mx = 0, sum = 0; size_t n = 0;
	void add(double d) { mx = std::max(mx, d); sum += d; ++n; }
	double mean() const { return n ? sum / n : 0; }
};

static int failures = 0;

static void verdict(const char *what, const char *clip, const Stat &s, double bound, bool mustPass, const char *unit,
		double scale) {
	bool pass = s.mx <= bound;
	bool ok = pass == mustPass;
	if (!ok) ++failures;
	printf("%s %-14s %-26s max %.6g mean %.6g %s (bound %.6g)  -> %s%s\n", ok ? "OK  " : "BAD ", clip, what,
			s.mx * scale, s.mean() * scale, unit, bound * scale, pass ? "PASS" : "FAIL",
			mustPass ? "" : " (control: must fail)");
}

static void distStat(const std::vector<float> &a, const std::vector<float> &b, Stat &st) {
	for (size_t i = 0; i + 2 < a.size(); i += 3)
		st.add(std::sqrt(double(a[i] - b[i]) * (a[i] - b[i]) + double(a[i + 1] - b[i + 1]) * (a[i + 1] - b[i + 1]) +
				double(a[i + 2] - b[i + 2]) * (a[i + 2] - b[i + 2])));
}

static int gpuMain(const std::string &dir, const std::string &gdir) {
	std::string m = slurp(dir + "/meta.json");
	size_t V = metaInt(m, "V"), S = 60;
	printf("GPU run (Godot RenderingDevice, windowed): %s\n", gdir.c_str());
	for (const char *clip : { "crawl_to_stand", "walk_to_stand" }) {
		std::string cd = dir + "/clips/" + clip + "/", gd = gdir + "/" + clip + "/";
		auto von = load<float>(cd + "verts_on.bin", S * 3 * V);
		const double bound = vertBound(von);
		Stat par, cOff, cTw, cShift, cKabsch;
		distStat(load<float>(gd + "verts.bin", S * 3 * V), von, par);
		distStat(load<float>(gd + "off.bin", S * 3 * V), von, cOff);
		distStat(load<float>(gd + "tw.bin", S * 3 * V), von, cTw);
		distStat(load<float>(gd + "shift.bin", S * 3 * V), von, cShift);
		distStat(load<float>(gd + "kabsch.bin", S * 3 * V), von, cKabsch);
		verdict("gpu vertices", clip, par, bound, true, "mm", 1000);
		verdict("gpu correctives off", clip, cOff, bound, false, "mm", 1000);
		verdict("gpu twist joints skipped", clip, cTw, bound, false, "mm", 1000);
		verdict("gpu one frame late", clip, cShift, bound, false, "mm", 1000);
		printf("INFO %-14s gpu, Kabsch swing-twist split (twistForm 0, the default): max %.4g mean %.4g mm from SOMA-X\n", clip,
				cKabsch.mx * 1000, cKabsch.mean() * 1000);
	}
	printf("%s\n", failures ? "FAILED" : "ALL OK");
	return failures ? 1 : 0;
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: %s <fixture dir> [--gpu <dir>]\n", argv[0]); return 2; }
	std::string dir = argv[1];
	if (argc >= 4 && std::string(argv[2]) == "--gpu") return gpuMain(dir, argv[3]);
	Rig r = loadRig(dir);
	printf("rig: %u public, %u target (%u twist), %u verts, correctives %u -> %u, W2 %u entries, FK rounds %u + %u\n",
			r.NJ, r.NT, r.NTW, r.V, r.D, r.K, r.E, r.roundsPub, r.roundsTgt);
	std::vector<float> zeroM(r.twistM.size(), 0.0f), zeroW2(r.w2val.size(), 0.0f);
	Frame f(r);
	const char *clips[] = { "crawl_to_stand", "walk_to_stand" };
	for (const char *clip : clips) {
		std::string cd = dir + "/clips/" + clip + "/";
		std::string cm = slurp(cd + "meta.json");
		size_t F = metaInt(cm, "F"), S = 60;
		auto pose = load<float>(cd + "pose.bin", F * 9 * r.NJ);
		auto transl = load<float>(cd + "transl.bin", F * 3);
		auto frames = load<uint32_t>(cd + "frames.bin", S);
		auto von = load<float>(cd + "verts_on.bin", S * 3 * r.V);
		auto wor = load<float>(cd + "world.bin", S * 12 * r.NT);
		Stat par, cOff, cTw, cShift, cKabsch, worldErr;
		const double VERT_BOUND_M = vertBound(von);
		double ms = 0;
		for (size_t s = 0; s < S; ++s) {
			size_t fr = frames[s];
			auto vs = [&](Stat &st) {
				for (uint32_t v = 0; v < r.V; ++v) {
					const float *a = &f.verts[3 * v], *b = &von[(s * r.V + v) * 3];
					st.add(std::sqrt(double(a[0] - b[0]) * (a[0] - b[0]) + double(a[1] - b[1]) * (a[1] - b[1]) +
							double(a[2] - b[2]) * (a[2] - b[2])));
				}
			};
			auto t0 = std::chrono::steady_clock::now();
			poseFrame(r, &pose[fr * 9 * r.NJ], &transl[fr * 3], r.twistM.data(), r.w2val.data(), f);
			ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			vs(par);
			for (size_t k = 0; k < 12 * r.NT; ++k) worldErr.add(std::fabs(double(f.world[k]) - wor[s * 12 * r.NT + k]) / std::max(1.0, std::fabs(double(wor[s * 12 * r.NT + k]))));
			poseFrame(r, &pose[fr * 9 * r.NJ], &transl[fr * 3], r.twistM.data(), zeroW2.data(), f);
			vs(cOff);
			poseFrame(r, &pose[fr * 9 * r.NJ], &transl[fr * 3], zeroM.data(), r.w2val.data(), f);
			vs(cTw);
			poseFrame(r, &pose[(fr + 1) * 9 * r.NJ], &transl[(fr + 1) * 3], r.twistM.data(), r.w2val.data(), f);
			vs(cShift);
			poseFrame(r, &pose[fr * 9 * r.NJ], &transl[fr * 3], r.twistM.data(), r.w2val.data(), f, 0);
			vs(cKabsch);
		}
		printf("INFO %-14s cpu %.2f ms/frame (single thread, C++ emission); target world max relative |d| %.3g\n",
				clip, ms / S, worldErr.mx);
		verdict("vertices", clip, par, VERT_BOUND_M, true, "mm", 1000);
		verdict("correctives off", clip, cOff, VERT_BOUND_M, false, "mm", 1000);
		verdict("twist joints skipped", clip, cTw, VERT_BOUND_M, false, "mm", 1000);
		verdict("one frame late", clip, cShift, VERT_BOUND_M, false, "mm", 1000);
		printf("INFO %-14s twist by the Kabsch swing-twist split (twistForm 0, the default) instead of SOMA-X's half-angle form: "
				"max %.4g mean %.4g mm from SOMA-X\n", clip, cKabsch.mx * 1000, cKabsch.mean() * 1000);
	}
	printf("%s\n", failures ? "FAILED" : "ALL OK");
	return failures ? 1 : 0;
}
