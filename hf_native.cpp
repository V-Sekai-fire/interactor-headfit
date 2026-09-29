// hf_native -- the head fit's native check runner (Gate 11, gates/11-headfit):
// guest/headfit/hf_core compiled for the host, driven through the same calls
// headfit.elf exposes, on ANNY's CC0 sources and the avatar fixture
// (tests/headfit/export_fixture.gd).
//
//   hf_native <mode> [key=value ...]
//
// modes: load, probe, marks, fit, transfer, identity, audit, coeffs, dump.
// keys:  anny=<ANNY data dir> dials=<local_dials.txt> map=<unified_expressions.map>
//        fixture=<dir with avatar_body.usda, avatar_shapes.txt>
//        marks=<file> out=<file or dir> required=<names file> cases=<phenotype list>
//        control=shuffle|corrupt-target|none
// SPDX-License-Identifier: Apache-2.0 OR MIT
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "hf_core.h"
#include "hf_io.h"

namespace {

std::map<std::string, std::string> g_kv;

std::string arg(const char *k, const std::string &def = "") {
	auto it = g_kv.find(k);
	return it == g_kv.end() ? def : it->second;
}

// A height band (mm) given as `key=lo,hi`, or the default.
double band(const char *k, int which, double lo, double hi) {
	const std::string v = arg(k);
	const size_t comma = v.find(',');
	if (comma == std::string::npos) {
		return which == 0 ? lo : hi;
	}
	return std::stod(which == 0 ? v.substr(0, comma) : v.substr(comma + 1));
}

int die(const std::string &why) {
	std::printf("FAIL %s\n", why.c_str());
	return 2;
}

struct Ctx {
	hfio::UploadStats up;
	int files = 0;
	int model = -1, mesh = -1, marks = -1, fit = -1;
	hfio::Avatar av;
	std::vector<hfio::Mark> mk;
};

bool upload(Ctx &c) {
	std::string err;
	c.files = hfio::upload_anny(arg("anny"), arg("dials"), arg("map"), c.up, err);
	if (c.files < 0) {
		std::printf("FAIL upload: %s\n", err.c_str());
		return false;
	}
	std::printf("upload files %d base %d macro %d dial %d facial %d unit %d text %d\n", c.files, c.up.base, c.up.macro,
			c.up.dial, c.up.facial, c.up.unit, c.up.text);
	return true;
}

// The gas budget per tick (budget=, default unlimited) and the ticks spent.
int64_t g_budget = 0;
uint64_t g_ticks = 0;

// Tick a job until DONE; events (non-RUNNING-by-gas lines) go to stdout when
// `echo` says so. False on FAIL.
bool run(int h, const char *what, bool echo, std::string *last = nullptr) {
	for (;;) {
		const std::string s = hf::tick(h, g_budget);
		++g_ticks;
		if (s.rfind("FAIL", 0) == 0) {
			std::printf("FAIL %s: %s\n", what, s.c_str());
			return false;
		}
		if (echo && (s.find(" start ") != std::string::npos || s.find(" done ") != std::string::npos ||
							arg("verbose") == "1")) {
			std::printf("tick %s\n", s.c_str());
		}
		if (s.rfind("DONE", 0) == 0) {
			if (last) {
				*last = s;
			}
			return true;
		}
	}
}

bool load_model(Ctx &c) {
	c.model = hf::model_load(0);
	if (c.model < 0) {
		std::printf("FAIL model_load: %s\n", hf::last_error().c_str());
		return false;
	}
	return run(c.model, "model_load", false);
}

bool load_avatar(Ctx &c, bool with_shapes) {
	std::string err;
	const std::string fx = arg("fixture");
	if (!hfio::read_avatar(fx + "/avatar_body.usda", fx + "/avatar_shapes.txt", c.av, err)) {
		std::printf("FAIL avatar: %s\n", err.c_str());
		return false;
	}
	c.mesh = hf::mesh_create(c.av.xyz.data(), int(c.av.xyz.size() / 3), c.av.tri.data(), int(c.av.tri.size() / 3));
	if (c.mesh < 0) {
		std::printf("FAIL mesh_create: %s\n", hf::last_error().c_str());
		return false;
	}
	if (with_shapes) {
		for (size_t s = 0; s < c.av.shapes.size(); ++s) {
			if (hf::mesh_add_shape(c.mesh, c.av.shape_names[s], c.av.shapes[s].data(), int(c.av.shapes[s].size())) < 0) {
				std::printf("FAIL mesh_add_shape %s: %s\n", c.av.shape_names[s].c_str(), hf::last_error().c_str());
				return false;
			}
		}
	}
	return true;
}

// ---- probe: landmark candidates from measurements ------------------------------

struct Pt {
	double x, y, z;
};

// Top vertices of a delta field by magnitude, per side of x (the mesh's own
// frame): side +1 x > xc + band, -1 x < xc - band, 0 |x - xc| <= band.
void top_by_side(const char *what, const std::vector<float> &xyz, const std::vector<float> &d, const std::vector<int32_t> *ids,
		double xc, double band, double unit) {
	const size_t n = xyz.size() / 3;
	for (int side : { +1, 0, -1 }) {
		std::vector<std::pair<double, size_t>> v;
		for (size_t i = 0; i < n; ++i) {
			const double x = xyz[i * 3] - xc;
			if ((side > 0 && x <= band) || (side < 0 && x >= -band) || (side == 0 && std::fabs(x) > band)) {
				continue;
			}
			const double m = std::sqrt(double(d[i * 3]) * d[i * 3] + double(d[i * 3 + 1]) * d[i * 3 + 1] + double(d[i * 3 + 2]) * d[i * 3 + 2]);
			if (m > 0) {
				v.push_back({ -m, i });
			}
		}
		std::sort(v.begin(), v.end());
		std::printf("probe %s side %+d moved %zu:", what, side, v.size());
		for (size_t k = 0; k < std::min<size_t>(3, v.size()); ++k) {
			const size_t i = v[k].second;
			std::printf(" [id %d |d| %.3f mm at %.1f %.1f %.1f]", ids ? (*ids)[i] : int(i), -v[k].first * unit,
					xyz[i * 3] * unit, xyz[i * 3 + 1] * unit, xyz[i * 3 + 2] * unit);
		}
		std::printf("\n");
	}
}

int mode_probe(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true)) {
		return 2;
	}
	const double ph[6] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<float> xyz;
	std::vector<int32_t> tri, gid;
	hf::model_mesh(c.model, ph, xyz, tri, gid);
	double lo[3] = { 1e9, 1e9, 1e9 }, hi[3] = { -1e9, -1e9, -1e9 };
	for (size_t i = 0; i < xyz.size() / 3; ++i) {
		for (int k = 0; k < 3; ++k) {
			lo[k] = std::min(lo[k], double(xyz[i * 3 + k])), hi[k] = std::max(hi[k], double(xyz[i * 3 + k]));
		}
	}
	std::printf("anny region %zu v, bounds mm x %.1f..%.1f y %.1f..%.1f z %.1f..%.1f\n", xyz.size() / 3, lo[0] * 1e3,
			hi[0] * 1e3, lo[1] * 1e3, hi[1] * 1e3, lo[2] * 1e3, hi[2] * 1e3);
	for (const char *a : { "browOuterUpLeft", "browInnerUp", "browDownLeft", "mouthStretchLeft", "mouthSmileLeft",
				 "mouthRollUpper", "mouthRollLower", "mouthPucker", "jawOpen", "eyeBlinkLeft", "noseSneerLeft" }) {
		std::vector<float> d;
		hf::model_action(c.model, a, d);
		top_by_side(a, xyz, d, &gid, 0.0, 0.0005, 1e3);
	}
	const std::vector<float> &ax = c.av.xyz;
	for (int k = 0; k < 3; ++k) {
		lo[k] = 1e9, hi[k] = -1e9;
	}
	for (size_t i = 0; i < ax.size() / 3; ++i) {
		for (int k = 0; k < 3; ++k) {
			lo[k] = std::min(lo[k], double(ax[i * 3 + k])), hi[k] = std::max(hi[k], double(ax[i * 3 + k]));
		}
	}
	std::printf("avatar %zu v %zu t, bounds mm x %.1f..%.1f y %.1f..%.1f z %.1f..%.1f\n", ax.size() / 3, c.av.tri.size() / 3,
			lo[0] * 1e3, hi[0] * 1e3, lo[1] * 1e3, hi[1] * 1e3, lo[2] * 1e3, hi[2] * 1e3);
	std::map<int, int> sm;
	std::map<int, std::pair<double, double>> sz;
	for (size_t f = 0; f < c.av.submesh.size(); ++f) {
		++sm[c.av.submesh[f]];
	}
	for (auto &kv : sm) {
		std::printf("avatar submesh %d tris %d\n", kv.first, kv.second);
	}
	for (size_t s = 0; s < c.av.shapes.size(); ++s) {
		top_by_side(c.av.shape_names[s].c_str(), ax, c.av.shapes[s], nullptr, 0.0, 0.0005, 1e3);
	}
	return 0;
}

// The frontmost midline vertex per 2 mm of height: id, height, forward.
// up/fwd are axis indices; fsign turns the forward axis to +.
void profile(const char *what, const std::vector<float> &xyz, const std::vector<int32_t> *ids, int up, int fwd,
		double fsign, double lo_mm, double hi_mm) {
	std::map<int, std::pair<double, size_t>> bin;
	for (size_t i = 0; i < xyz.size() / 3; ++i) {
		if (std::fabs(xyz[i * 3]) > 0.0003) {
			continue;
		}
		const double h = xyz[i * 3 + up] * 1e3, f = fsign * xyz[i * 3 + fwd] * 1e3;
		if (h < lo_mm || h > hi_mm) {
			continue;
		}
		const int b = int(std::floor(h / 2.0));
		auto it = bin.find(b);
		if (it == bin.end() || f > it->second.first) {
			bin[b] = { f, i };
		}
	}
	for (auto it = bin.rbegin(); it != bin.rend(); ++it) {
		const size_t i = it->second.second;
		std::printf("profile %s id %d up %.1f fwd %.1f\n", what, ids ? (*ids)[i] : int(i), xyz[i * 3 + up] * 1e3,
				it->second.first);
	}
}

int mode_profile(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true)) {
		return 2;
	}
	const double ph[6] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<float> xyz;
	std::vector<int32_t> tri, gid;
	hf::model_mesh(c.model, ph, xyz, tri, gid);
	profile("anny", xyz, &gid, 1, 2, 1.0, 560.0, 740.0);
	profile("avatar", c.av.xyz, nullptr, 2, 1, -1.0, band("profile_band", 0, 1455.0, 1555.0),
			band("profile_band", 1, 1455.0, 1555.0));
	return 0;
}

// ---- marks: the landmark pairs, by one rule on both meshes ----------------------
//
// Each mesh in its own frame: ANNY x left, y up, z forward (base.obj); the
// avatar x left, z up, -y forward (the fixture, mirrored to Godot). The rules
// read shapes of the same meaning on both: a sideways mouth stretch, a jaw
// opening, an inner brow raise, an outer brow raise.

struct Frame {
	const std::vector<float> *xyz;
	int up, fwd;
	double fsign;
	double h(size_t i) const { return (*xyz)[i * 3 + up] * 1e3; }
	double f(size_t i) const { return fsign * (*xyz)[i * 3 + fwd] * 1e3; }
	double x(size_t i) const { return (*xyz)[i * 3] * 1e3; }
	size_t n() const { return xyz->size() / 3; }
};

double mag(const std::vector<float> &d, size_t i) {
	return std::sqrt(double(d[i * 3]) * d[i * 3] + double(d[i * 3 + 1]) * d[i * 3 + 1] + double(d[i * 3 + 2]) * d[i * 3 + 2]);
}

// argmax |d| on one side (+1: x > 0.5 mm, -1: x < -0.5 mm); ties: lower index.
long argmax_side(const Frame &F, const std::vector<float> &d, int side) {
	long best = -1;
	double bm = 0.0;
	for (size_t i = 0; i < F.n(); ++i) {
		if (side * F.x(i) <= 0.5) {
			continue;
		}
		const double m = mag(d, i);
		if (m > bm) {
			bm = m, best = long(i);
		}
	}
	return best;
}

// The extreme-|x| vertex of a shape's support (|d| > 0.1 mm) on one side.
long outermost_side(const Frame &F, const std::vector<float> &d, int side) {
	long best = -1;
	double bx = 0.0;
	for (size_t i = 0; i < F.n(); ++i) {
		if (side * F.x(i) <= 0.5 || mag(d, i) * 1e3 <= 0.1) {
			continue;
		}
		if (side * F.x(i) > bx) {
			bx = side * F.x(i), best = long(i);
		}
	}
	return best;
}

// The outermost vertex on a side among those moving at least 90% of the shape's most: a mouth-widen
// sculpt that moves the lip ring as one ties on the strongest vertex, and the corner is its edge.
long outermost_strong_side(const Frame &F, const std::vector<float> &d, int side) {
	double top = 0.0;
	for (size_t i = 0; i < F.n(); ++i) {
		if (side * F.x(i) > 0.5) {
			top = std::max(top, mag(d, i));
		}
	}
	long best = -1;
	double bx = 0.0;
	for (size_t i = 0; i < F.n(); ++i) {
		if (side * F.x(i) > 0.5 && mag(d, i) >= 0.9 * top && side * F.x(i) > bx) {
			bx = side * F.x(i), best = long(i);
		}
	}
	return best;
}

// The front midline profile between two heights: midline vertices (|x| <
// 0.3 mm) on the front surface -- none within 1.5 mm of height stands 2 mm
// further forward (the mouth's interior drops out) -- the frontmost per 1 mm
// of height, top to bottom.
std::vector<size_t> front_profile(const Frame &F, double lo, double hi) {
	std::vector<size_t> mid;
	for (size_t i = 0; i < F.n(); ++i) {
		if (std::fabs(F.x(i)) < 0.3 && F.h(i) >= lo && F.h(i) <= hi) {
			mid.push_back(i);
		}
	}
	std::map<int, size_t> bin;
	for (size_t i : mid) {
		bool front = true;
		for (size_t j : mid) {
			if (std::fabs(F.h(j) - F.h(i)) < 1.5 && F.f(j) > F.f(i) + 2.0) {
				front = false;
				break;
			}
		}
		if (!front) {
			continue;
		}
		const int b = int(std::floor(F.h(i)));
		auto it = bin.find(b);
		if (it == bin.end() || F.f(i) > F.f(it->second) || (F.f(i) == F.f(it->second) && i < it->second)) {
			bin[b] = i;
		}
	}
	std::vector<size_t> p;
	for (auto it = bin.rbegin(); it != bin.rend(); ++it) {
		p.push_back(it->second);
	}
	return p;
}

struct MarkSet {
	long corner_l = -1, corner_r = -1, lip_upper = -1, lip_lower = -1, chin = -1;
	long brow_in_l = -1, brow_in_r = -1, brow_out_l = -1, brow_out_r = -1;
	std::string how;
};

// open: the jaw-opening shape; the profile window spans nose to chin.
void mark_mouth(const Frame &F, const std::vector<float> &open, double lo, double hi, MarkSet &m) {
	const std::vector<size_t> p = front_profile(F, lo, hi);
	double omax = 0.0;
	for (size_t i : p) {
		omax = std::max(omax, mag(open, i));
	}
	size_t k = 0;
	while (k < p.size() && mag(open, p[k]) <= 0.5 * omax) {
		++k;
	}
	if (k == 0 || k >= p.size()) {
		return;
	}
	m.lip_lower = long(p[k]);
	m.lip_upper = long(p[k - 1]);
	// The chin (gnathion): below the labiomental fold (the least forward
	// profile vertex 5..25 mm under the lower lip) and the pogonion (the most
	// forward one below the fold), the first profile step that turns 45
	// degrees under (d fwd / d height >= 1). A chin that never turns (the mesh
	// ends first) is its lowest profile vertex.
	const double hl = F.h(p[k]);
	size_t fold = 0;
	for (size_t j = k + 1; j < p.size(); ++j) {
		if (F.h(p[j]) <= hl - 5.0 && F.h(p[j]) >= hl - 25.0 && (fold == 0 || F.f(p[j]) < F.f(p[fold]))) {
			fold = j;
		}
	}
	size_t pog = fold;
	for (size_t j = fold + 1; fold && j < p.size(); ++j) {
		if (F.f(p[j]) > F.f(p[pog])) {
			pog = j;
		}
	}
	for (size_t j = pog + 1; pog && j < p.size(); ++j) {
		const double dh = F.h(p[j - 1]) - F.h(p[j]), df = F.f(p[j - 1]) - F.f(p[j]);
		if (dh > 0 && df / dh >= 1.0) {
			m.chin = long(p[j]);
			m.how += " chin=45deg-below-pogonion";
			return;
		}
	}
	m.chin = long(p.back());
	m.how += " chin=lowest";
}

int mode_marks(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true)) {
		return 2;
	}
	const double ph[6] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<float> axyz;
	std::vector<int32_t> tri, gid;
	hf::model_mesh(c.model, ph, axyz, tri, gid);
	auto act = [&](const char *a) {
		std::vector<float> d;
		hf::model_action(c.model, a, d);
		return d;
	};
	auto shape = [&](const char *s) -> const std::vector<float> & {
		for (size_t k = 0; k < c.av.shape_names.size(); ++k) {
			if (c.av.shape_names[k] == s) {
				return c.av.shapes[k];
			}
		}
		static std::vector<float> none;
		std::printf("FAIL the fixture has no shape %s\n", s);
		std::exit(2);
		return none;
	};
	const Frame FA{ &axyz, 1, 2, 1.0 }, FV{ &c.av.xyz, 2, 1, -1.0 };
	MarkSet A, V;
	A.corner_l = argmax_side(FA, act("mouthStretchLeft"), +1);
	A.corner_r = argmax_side(FA, act("mouthStretchRight"), -1);
	// The avatar's shapes that mark the mouth corners, the open mouth and the brows.
	const std::string wide = arg("shape_wide", "mouth_wide"), open = arg("shape_open", "vrc.v_aa");
	const std::string brow_in = arg("shape_brow_in", "brow_trouble"), brow_out = arg("shape_brow_out", "brow_up");
	V.corner_l = outermost_strong_side(FV, shape(wide.c_str()), +1);
	V.corner_r = outermost_strong_side(FV, shape(wide.c_str()), -1);
	// Nose tip to below the chin: ANNY 560..660 mm, the avatar's mouth_band (Mire's 1455..1535 mm).
	mark_mouth(FA, act("jawOpen"), 560.0, 660.0, A);
	mark_mouth(FV, shape(open.c_str()), band("mouth_band", 0, 1455.0, 1535.0), band("mouth_band", 1, 1455.0, 1535.0), V);
	A.brow_in_l = argmax_side(FA, act("browInnerUp"), +1);
	A.brow_in_r = argmax_side(FA, act("browInnerUp"), -1);
	V.brow_in_l = argmax_side(FV, shape(brow_in.c_str()), +1);
	V.brow_in_r = argmax_side(FV, shape(brow_in.c_str()), -1);
	A.brow_out_l = argmax_side(FA, act("browOuterUpLeft"), +1);
	A.brow_out_r = argmax_side(FA, act("browOuterUpRight"), -1);
	V.brow_out_l = outermost_side(FV, shape(brow_out.c_str()), +1);
	V.brow_out_r = outermost_side(FV, shape(brow_out.c_str()), -1);
	struct Row {
		const char *name;
		long a, v;
	};
	const Row rows[] = { { "mouth_corner_l", A.corner_l, V.corner_l }, { "mouth_corner_r", A.corner_r, V.corner_r },
		{ "lip_mid_upper", A.lip_upper, V.lip_upper }, { "lip_mid_lower", A.lip_lower, V.lip_lower },
		{ "chin", A.chin, V.chin }, { "brow_inner_l", A.brow_in_l, V.brow_in_l },
		{ "brow_inner_r", A.brow_in_r, V.brow_in_r }, { "brow_outer_l", A.brow_out_l, V.brow_out_l },
		{ "brow_outer_r", A.brow_out_r, V.brow_out_r } };
	std::string out = "# name model_id(base.obj, 0-based) avatar_id -- tests/headfit/hf_native marks\n";
	for (const Row &r : rows) {
		if (r.a < 0 || r.v < 0) {
			std::printf("FAIL mark %s not found (anny %ld avatar %ld)\n", r.name, r.a, r.v);
			return 1;
		}
		std::printf("mark %-15s anny %5d (x %6.1f up %6.1f fwd %6.1f mm)  avatar %5ld (x %6.1f up %7.1f fwd %6.1f mm)\n",
				r.name, gid[size_t(r.a)], FA.x(size_t(r.a)), FA.h(size_t(r.a)), FA.f(size_t(r.a)), r.v, FV.x(size_t(r.v)),
				FV.h(size_t(r.v)), FV.f(size_t(r.v)));
		out += std::string(r.name) + " " + std::to_string(gid[size_t(r.a)]) + " " + std::to_string(r.v) + "\n";
	}
	std::printf("rules anny%s avatar%s\n", A.how.c_str(), V.how.c_str());
	const std::string path = arg("out");
	if (!path.empty()) {
		FILE *f = std::fopen(path.c_str(), "wb");
		if (!f) {
			return die("cannot write " + path);
		}
		std::fwrite(out.data(), 1, out.size(), f);
		std::fclose(f);
		std::printf("wrote %s\n", path.c_str());
	}
	return 0;
}

// ---- fit ------------------------------------------------------------------------

bool load_marks(Ctx &c) {
	std::string err;
	if (!hfio::read_marks(arg("marks"), c.mk, err)) {
		std::printf("FAIL marks: %s\n", err.c_str());
		return false;
	}
	std::vector<int32_t> m, a;
	for (const hfio::Mark &k : c.mk) {
		m.push_back(k.model), a.push_back(k.avatar);
	}
	if (arg("control") == "shuffle") {
		// The planted control: the left mouth corner's avatar vertex swapped
		// with the right outer brow's.
		size_t i = 0, j = 0;
		for (size_t k = 0; k < c.mk.size(); ++k) {
			i = c.mk[k].name == "mouth_corner_l" ? k : i;
			j = c.mk[k].name == "brow_outer_r" ? k : j;
		}
		std::swap(a[i], a[j]);
		std::printf("control shuffle: %s <-> %s avatar ids\n", c.mk[i].name.c_str(), c.mk[j].name.c_str());
	}
	c.marks = hf::marks_create(m.data(), a.data(), int(m.size()));
	if (c.marks < 0) {
		std::printf("FAIL marks_create: %s\n", hf::last_error().c_str());
		return false;
	}
	return true;
}

// Weights from `w=a,b,c,...` (hf_solve's order), else the defaults.
std::vector<float> weights() {
	std::vector<float> w(hf::kWeightDefaults, hf::kWeightDefaults + hf::W_COUNT);
	const std::string s = arg("w");
	size_t k = 0, p = 0;
	while (!s.empty() && p <= s.size() && k < w.size()) {
		const size_t e = s.find(',', p);
		const std::string t = s.substr(p, e == std::string::npos ? std::string::npos : e - p);
		if (!t.empty()) {
			w[k] = float(std::atof(t.c_str()));
		}
		++k;
		if (e == std::string::npos) {
			break;
		}
		p = e + 1;
	}
	return w;
}

bool run_fit(Ctx &c, bool verbose) {
	const std::vector<float> w = weights();
	c.fit = hf::solve(c.model, c.mesh, c.marks, w.data(), int(w.size()));
	if (c.fit < 0) {
		std::printf("FAIL solve: %s\n", hf::last_error().c_str());
		return false;
	}
	return run(c.fit, "fit", verbose || true);
}

int mode_fit(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true) || !load_marks(c)) {
		return 2;
	}
	if (!run_fit(c, arg("verbose") == "1")) {
		return 1;
	}
	std::printf("%s", hf::fit_report(c.fit).c_str());
	std::printf("ticks %llu budget %lld\n", (unsigned long long)g_ticks, (long long)g_budget);
	std::vector<float> x;
	hf::fit_x(c.fit, x);
	std::string hex;
	for (float v : x) {
		unsigned char b[4];
		std::memcpy(b, &v, 4);
		char t[9];
		std::snprintf(t, sizeof t, "%02x%02x%02x%02x", b[0], b[1], b[2], b[3]);
		hex += t;
	}
	std::printf("xhex %s\n", hex.c_str()); // x's float32 bytes, little-endian (gate_headfit.gd prints the same)
	return 0;
}

// ---- dump: round 0's problem for the LBFGSpp oracle (tests/headfit_oracle) -----

template <class T>
bool write_bin(const std::string &path, const std::vector<T> &v) {
	FILE *f = std::fopen(path.c_str(), "wb");
	if (!f) {
		return false;
	}
	std::fwrite(v.data(), sizeof(T), v.size(), f);
	std::fclose(f);
	return true;
}

int mode_dump(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true) || !load_marks(c)) {
		return 2;
	}
	// One round: the problem the fit set up (after the landmark start), the
	// guest's L-BFGS-B result on it, and its own loss and gradient at x0.
	std::vector<float> w = weights();
	w[hf::W_ROUNDS] = 1.0f;
	c.fit = hf::solve(c.model, c.mesh, c.marks, w.data(), int(w.size()));
	if (c.fit < 0 || !run(c.fit, "fit", true)) {
		return 1;
	}
	hf::RoundProblem rp;
	std::vector<float> xg, g0;
	double fg = 0, f0 = 0;
	int iters = 0;
	if (!hf::round_problem(c.fit, 0, rp) || !hf::round_result(c.fit, 0, xg, fg, iters) ||
			!hf::evaluate_at(c.fit, rp.x0, f0, g0)) {
		return die(hf::last_error());
	}
	const std::string out = arg("out");
	std::string meta = "headfit-round 1\n";
	char b[256];
	std::snprintf(b, sizeof b, "P %u NB %u K %u n %u L %u macro_rows %u\n", rp.P, rp.NB, rp.K, rp.n, rp.L, rp.macro_rows);
	meta += b;
	std::snprintf(b, sizeof b, "lbfgsb m 6 epsilon 1e-4 epsilon_rel 0 past 1 delta 1e-8 max_iterations %d max_linesearch 30\n",
			std::max(1, int(w[hf::W_ITERS])));
	meta += b;
	meta += "tscale 100 (x[6..8] is the translation in units of 100 mm; the kernels take x * 100)\n";
	meta += "pscale 0.1 (x[10..15] is the phenotype in units of 0.1; the anchor rule takes x * 0.1)\n";
	std::snprintf(b, sizeof b, "guest f0 %.17g iters %d f %.17g\n", f0, iters, fg);
	meta += b;
	for (size_t a = 0; a < rp.anchors.size(); ++a) {
		meta += "axis";
		for (int s : rp.axis_slots[a]) {
			meta += " " + std::to_string(s);
		}
		meta += " anchors";
		for (double v : rp.anchors[a]) {
			std::snprintf(b, sizeof b, " %.17g", v);
			meta += b;
		}
		meta += "\n";
	}
	for (const auto &sl : rp.row_slots) {
		meta += "row";
		for (int s : sl) {
			meta += " " + std::to_string(s);
		}
		meta += "\n";
	}
	FILE *f = std::fopen((out + "/meta.txt").c_str(), "wb");
	if (!f) {
		return die("cannot write " + out + "/meta.txt");
	}
	std::fwrite(meta.data(), 1, meta.size(), f);
	std::fclose(f);
	bool ok = write_bin(out + "/base.f32", rp.base) && write_bin(out + "/blend.f32", rp.blend) &&
			write_bin(out + "/q.f32", rp.q) && write_bin(out + "/nrm.f32", rp.nrm) && write_bin(out + "/w.f32", rp.w) &&
			write_bin(out + "/mid.u32", rp.mid) && write_bin(out + "/tgt.f32", rp.tgt) && write_bin(out + "/lw.f32", rp.lw) &&
			write_bin(out + "/x0.f32", rp.x0) && write_bin(out + "/lb.f32", rp.lb) && write_bin(out + "/ub.f32", rp.ub) &&
			write_bin(out + "/px0.f32", rp.prior_x0) && write_bin(out + "/plam.f32", rp.prior_lam) &&
			write_bin(out + "/xg.f32", xg) && write_bin(out + "/g0.f32", g0);
	if (!ok) {
		return die("cannot write the dump");
	}
	std::printf("dump %s P %u NB %u n %u guest f0 %.9g f %.9g iters %d\n", out.c_str(), rp.P, rp.NB, rp.n, f0, fg, iters);
	return 0;
}

// ---- transfer: the 52 actions and every name of the map ---------------------------

const char *const kActions[52] = { "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
	"browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
	"eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
	"eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
	"jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
	"mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
	"mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
	"mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
	"mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut" };

// Transfer one shape; false with the refusal printed.
bool one_transfer(int fit, const std::string &shape, std::vector<float> &d, float &err, std::string &info,
		std::string &why) {
	const int job = hf::transfer(fit, shape);
	if (job < 0) {
		why = hf::last_error();
		return false;
	}
	std::string last;
	for (;;) {
		const std::string s = hf::tick(job, g_budget);
		++g_ticks;
		if (s.rfind("FAIL", 0) == 0) {
			why = s.substr(5);
			hf::destroy(job);
			return false;
		}
		if (s.rfind("DONE", 0) == 0) {
			break;
		}
	}
	const bool ok = hf::transfer_result(job, d, err, info);
	if (!ok) {
		why = hf::last_error();
	}
	hf::destroy(job);
	return ok;
}

std::vector<std::string> map_names(const std::string &path) {
	std::string t;
	hfio::read_file(path, t);
	std::vector<std::string> out;
	size_t p = 0;
	while (p < t.size()) {
		size_t e = t.find('\n', p);
		if (e == std::string::npos) {
			e = t.size();
		}
		std::string line = t.substr(p, e - p);
		p = e + 1;
		const size_t h = line.find('#');
		if (h != std::string::npos) {
			line = line.substr(0, h);
		}
		const size_t a = line.find("<=");
		if (a == std::string::npos) {
			continue;
		}
		std::string n = line.substr(0, a);
		while (!n.empty() && (n.back() == ' ' || n.back() == '\t')) {
			n.pop_back();
		}
		while (!n.empty() && (n[0] == ' ' || n[0] == '\t')) {
			n.erase(0, 1);
		}
		out.push_back(n);
	}
	return out;
}

int mode_transfer(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true) || !load_marks(c) || !run_fit(c, false)) {
		return 2;
	}
	std::printf("%s", hf::fit_report(c.fit).c_str());
	int refused52 = 0;
	for (const char *a : kActions) {
		std::vector<float> d;
		float err = 0;
		std::string info, why;
		if (one_transfer(c.fit, a, d, err, info, why)) {
			std::printf("action %s\n", info.c_str());
		} else {
			++refused52;
			std::printf("action_refused %s %s\n", a, why.c_str());
		}
	}
	std::printf("actions 52 transferred %d refused %d\n", 52 - refused52, refused52);
	std::map<std::string, std::vector<float>> got;
	int ok = 0, refused = 0;
	const std::vector<std::string> names = map_names(arg("map"));
	for (const std::string &n : names) {
		std::vector<float> d;
		float err = 0;
		std::string info, why;
		if (one_transfer(c.fit, n, d, err, info, why)) {
			++ok;
			std::printf("name %s\n", info.c_str());
			got[n] = d;
		} else {
			++refused;
			std::printf("name_refused %s %s\n", n.c_str(), why.c_str());
		}
	}
	std::printf("names %zu transferred %d refused %d\n", names.size(), ok, refused);
	// Split pairs: X{Left,Right} and X{Upper,Lower} sum to X within float precision.
	int pairs = 0, pairs_bad = 0;
	for (const auto &kv : got) {
		for (const auto &sd : { std::make_pair(std::string("Left"), std::string("Right")),
					 std::make_pair(std::string("Upper"), std::string("Lower")) }) {
			const std::string &n = kv.first;
			if (n.size() <= sd.first.size() || n.compare(n.size() - sd.first.size(), sd.first.size(), sd.first) != 0) {
				continue;
			}
			const std::string stem = n.substr(0, n.size() - sd.first.size());
			auto r = got.find(stem + sd.second);
			auto p = got.find(stem);
			if (r == got.end() || p == got.end()) {
				continue;
			}
			double worst = 0, mag = 0;
			for (size_t k = 0; k < p->second.size(); ++k) {
				worst = std::max(worst, std::fabs(double(kv.second[k]) + r->second[k] - p->second[k]));
				mag = std::max(mag, std::fabs(double(p->second[k])));
			}
			++pairs;
			const bool good = worst <= 1e-6 * std::max(mag, 1e-3) + 1e-9;
			pairs_bad += !good;
			std::printf("split %s = %s + %s max|sum - parent| %.3e mm (parent max %.4f mm) %s\n", stem.c_str(), n.c_str(),
					(stem + sd.second).c_str(), worst * 1e3, mag * 1e3, good ? "ok" : "OVER");
		}
	}
	std::printf("split pairs %d over %d\n", pairs, pairs_bad);
	return 0;
}

// ---- identity: ANNY onto itself reproduces its shapes -------------------------------

int mode_identity(Ctx &c) {
	if (!upload(c) || !load_model(c)) {
		return 2;
	}
	const double ph[6] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	std::vector<float> xyz;
	std::vector<int32_t> tri, gid;
	hf::model_mesh(c.model, ph, xyz, tri, gid);
	c.mesh = hf::mesh_create(xyz.data(), int(xyz.size() / 3), tri.data(), int(tri.size() / 3));
	if (c.mesh < 0) {
		return die(hf::last_error());
	}
	// The same nine marks the avatar uses, on ANNY's own vertices.
	std::string err;
	std::vector<hfio::Mark> mk;
	if (!hfio::read_marks(arg("marks"), mk, err)) {
		return die(err);
	}
	std::map<int32_t, int32_t> at;
	for (size_t i = 0; i < gid.size(); ++i) {
		at[gid[i]] = int32_t(i);
	}
	std::vector<int32_t> m, a;
	for (const hfio::Mark &k : mk) {
		m.push_back(k.model), a.push_back(at[k.model]);
	}
	c.marks = hf::marks_create(m.data(), a.data(), int(m.size()));
	if (c.marks < 0 || !run_fit(c, false)) {
		return 2;
	}
	std::printf("%s", hf::fit_report(c.fit).c_str());
	double worst_all = 0, worst_rel = 0;
	int over = 0;
	int refused_empty = 0;
	for (const char *act : kActions) {
		std::vector<float> d, ref;
		float e = 0;
		std::string info, why;
		hf::model_action(c.model, act, ref);
		if (!one_transfer(c.fit, act, d, e, info, why)) {
			// No support in the body region (tongueOut: the tongue is a helper
			// group, RFD 2275's next stage) is the right answer; any other
			// refusal is a failure.
			const bool empty = why.find("empty support (0 model vertices") != std::string::npos;
			std::printf("identity %s refused%s %s\n", act, empty ? "_empty" : "", why.c_str());
			refused_empty += empty;
			over += !empty;
			continue;
		}
		double worst = 0, mag = 0;
		for (size_t k = 0; k < ref.size(); ++k) {
			worst = std::max(worst, std::fabs(double(d[k]) - ref[k]));
			mag = std::max(mag, std::fabs(double(ref[k])));
		}
		const double rel = mag > 0 ? worst / mag : 0;
		worst_all = std::max(worst_all, worst), worst_rel = std::max(worst_rel, rel);
		const bool ok = rel <= 1e-2;
		over += !ok;
		std::printf("identity %-20s max|transferred - own| %.4f mm of %.4f mm (rel %.2e) %s\n", act, worst * 1e3, mag * 1e3,
				rel, ok ? "ok" : "OVER");
	}
	std::printf("identity 52 worst %.4f mm worst_rel %.3e refused_empty %d over %d\n", worst_all * 1e3, worst_rel,
			refused_empty, over);
	return over ? 1 : 0;
}

// ---- audit: the map against the required names ------------------------------------

int mode_audit(Ctx &c) {
	if (!upload(c) || !load_model(c) || !load_avatar(c, true)) {
		return 2;
	}
	std::string req;
	if (!hfio::read_file(arg("required"), req)) {
		return die("cannot read required=" + arg("required"));
	}
	std::printf("%s", hf::map_audit(req, c.mesh).c_str());
	return 0;
}

// ---- coeffs: the anchor rule at given phenotypes (the parity check) ------------------

int mode_coeffs(Ctx &c) {
	if (!upload(c) || !load_model(c)) {
		return 2;
	}
	std::string t;
	if (!hfio::read_file(arg("cases"), t)) {
		return die("cannot read cases=" + arg("cases"));
	}
	size_t p = 0;
	int k = 0;
	while (p < t.size()) {
		size_t e = t.find('\n', p);
		if (e == std::string::npos) {
			e = t.size();
		}
		const std::string line = t.substr(p, e - p);
		p = e + 1;
		double ph[6];
		if (line.empty() || line[0] == '#' ||
				std::sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf", &ph[0], &ph[1], &ph[2], &ph[3], &ph[4], &ph[5]) != 6) {
			continue;
		}
		std::vector<std::string> names;
		std::vector<double> cf;
		if (!hf::macro_coefficients(c.model, ph, names, cf)) {
			return die(hf::last_error());
		}
		for (size_t r = 0; r < names.size(); ++r) {
			std::printf("case %d %s %.9g\n", k, names[r].c_str(), cf[r]);
		}
		++k;
	}
	return 0;
}

// ---- load: counts and checksums -------------------------------------------------

int mode_load(Ctx &c) {
	if (!upload(c) || !load_model(c)) {
		return 2;
	}
	std::printf("%s", hf::data_report().c_str());
	std::printf("%s", hf::model_report(c.model).c_str());
	return 0;
}

} // namespace

int main(int argc, char **argv) {
	if (argc < 2) {
		std::printf("usage: hf_native <mode> [key=value ...]\n");
		return 2;
	}
	const std::string mode = argv[1];
	for (int k = 2; k < argc; ++k) {
		const char *e = std::strchr(argv[k], '=');
		if (e) {
			g_kv[std::string(argv[k], size_t(e - argv[k]))] = e + 1;
		}
	}
	Ctx c;
	if (arg("control") == "corrupt-target") {
		hfio::set_corrupt("faceunits01/targets/faceunits/jawOpen.target");
		std::printf("control corrupt-target: jawOpen.target's first delta moved by 1 dm\n");
	}
	g_budget = arg("budget").empty() ? std::numeric_limits<int64_t>::max() : std::atoll(arg("budget").c_str());
	if (mode == "load") {
		return mode_load(c);
	}
	if (mode == "transfer") {
		return mode_transfer(c);
	}
	if (mode == "identity") {
		return mode_identity(c);
	}
	if (mode == "audit") {
		return mode_audit(c);
	}
	if (mode == "coeffs") {
		return mode_coeffs(c);
	}
	if (mode == "dump") {
		return mode_dump(c);
	}
	if (mode == "fit") {
		return mode_fit(c);
	}
	if (mode == "marks") {
		return mode_marks(c);
	}
	if (mode == "profile") {
		return mode_profile(c);
	}
	if (mode == "probe") {
		return mode_probe(c);
	}
	return die("unknown mode " + mode);
}
