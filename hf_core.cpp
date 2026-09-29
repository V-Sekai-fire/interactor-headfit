// SPDX-License-Identifier: Apache-2.0 OR MIT
#include "hf_core.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

#include "hf_geom.h"
#include "hf_kernels.h"
#include "lbfgsb.h"
#include "similarity.h"
#include "vec_cpu.h"

namespace hf {

const float kWeightDefaults[W_COUNT] = {
	1.0f, // W_FACE
	1.0f, // W_MOUTH
	0.2f, // W_NOSE: down-weighted (RFD 2275)
	0.0f, // W_EYES: masked (RFD 2275)
	0.5f, // W_CRANIUM
	1.0f, // W_LANDMARK
	1.0f, // W_PRIOR_PHEN
	1.0f, // W_PRIOR_DIAL
	0.1f, // W_POINT
	6.0f, // W_ROUNDS
	60.0f, // W_ITERS
	8.0f, // W_MAXDIST (mm)
};

const char *region_name(int r) {
	static const char *n[R_COUNT] = { "all", "face", "mouth", "nose", "eyes", "cranium", "neck" };
	return r >= 0 && r < R_COUNT ? n[r] : "?";
}

namespace {

std::string g_err;

bool fail(const char *f, ...) {
	char b[1024];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	g_err = b;
	return false;
}

std::string fmt(const char *f, ...) {
	char b[2048];
	va_list ap;
	va_start(ap, f);
	std::vsnprintf(b, sizeof b, f, ap);
	va_end(ap);
	return b;
}

// MPFB2's units are decimetres; the fit's are millimetres.
// The neck band below the head line (the joint-neck cube's centroid): 0.3 dm
// reaches neck-platysma's lowest moved vertex (5.667 dm on base.obj, the
// head line 5.890 dm); the model report counts the unit entries it misses.
constexpr double kNeckBandDm = 0.3;
constexpr double kDmToMm = 100.0;
constexpr double kMToMm = 1000.0;
// x[6..8] holds the translation in units of kTScale mm, so a unit step of
// every rigid unknown moves the surface by about as much (rot6 and s by about
// the head's radius): the float32 L-BFGS-B stalled in the line search on
// the 100:1 mix (gates/11-headfit, the oracle). The kernels get prm, the
// translation in mm.
constexpr float kTScale = 100.0f;
// x[10..15] holds the phenotype in units of kPScale: a unit step moves the
// head by tens of mm, as the rigid unknowns do (unscaled, its gradient was
// 40x the dials' and 25x s's, and float32 L-BFGS-B curvature pairs lost it).
constexpr float kPScale = 0.1f;
void phen_of(const float *x, float ph[6]) {
	for (int a = 0; a < 6; ++a) {
		ph[a] = x[10 + a] * kPScale;
	}
}
void prm_of(const float *x, float prm[10]) {
	for (int k = 0; k < 10; ++k) {
		prm[k] = k >= 6 && k < 9 ? x[k] * kTScale : x[k];
	}
}

// ---- gas ------------------------------------------------------------------------
//
// Every job (the loader, a model, a fit, a transfer) advances in items, and a
// tick spends a budget of units on them: at least one item, then items while
// the budget lasts. A unit is about 1024 inner-loop iterations. Items never
// split an arithmetic sequence differently by budget, so any slicing gives
// the same bits (the gate runs budget 1 against an unlimited one).

struct Gas {
	int64_t left;
	int64_t items = 0;
	bool more() const { return items == 0 || left > 0; }
	void spend(int64_t cost) {
		left -= std::max<int64_t>(cost, 1);
		++items;
	}
};

int64_t kilo(uint64_t n) {
	return int64_t(n / 1024) + 1;
}

// ---- ANNY's sources: a queue parsed in slices ----------------------------------

std::map<std::string, std::string> &raw() {
	static std::map<std::string, std::string> d;
	return d;
}

struct Base {
	bool ok = false;
	uint32_t nv = 0, nfaces = 0, nquads = 0, ntris = 0;
	std::vector<double> v; // nv*3, dm
	std::vector<uint32_t> body_tris; // group "body", quads split (a,b,c) (a,c,d)
	double y_neck = 0.0; // the joint-neck cube's centroid height (dm)
	uint32_t neck_cube_verts = 0;
	std::vector<int32_t> cand; // global -> candidate (body, y >= y_neck - kNeckBandDm), or -1
	std::vector<uint32_t> cand_ids; // candidate -> global
};
Base g_base;
uint64_t g_base_gen = 0; // bumped by each parsed base.obj: models built on another are stale

// A target file as parsed: its whole-file checksums (the load gate compares
// them with an independent reader) and its entries on the candidate region.
struct TFile {
	uint64_t lines = 0;
	int64_t idx_sum = 0;
	int64_t idx_max = -1;
	double sum = 0.0, abs_sum = 0.0;
	uint32_t dropped_moving = 0; // entries off the candidates that move > 0.1 mm
	std::vector<uint32_t> ci; // candidate indices
	std::vector<float> d; // 3 per entry, mm
};
std::map<std::string, TFile> &tfiles() {
	static std::map<std::string, TFile> t;
	return t;
}

bool is_target_key(const std::string &k) {
	return k.size() > 7 && k.compare(k.size() - 7, 7, ".target") == 0;
}
const char *const kBaseKey = "mpfb2/3dobjs/base.obj";

// An upload being parsed: the text, where the parse stands, what it built.
struct Parse {
	std::string key, text;
	size_t pos = 0;
	uint64_t lineno = 0;
	bool base = false;
	Base b;
	std::string group;
	std::vector<uint32_t> neck;
	TFile t;
};
std::deque<Parse> &parse_queue() {
	static std::deque<Parse> q;
	return q;
}
std::string g_load_err; // the first parse failure since data_clear
uint64_t g_parsed = 0;
constexpr uint32_t kLinesPerItem = 1024;

bool base_line(Parse &p, std::string line) {
	Base &b = p.b;
	if (!line.empty() && line.back() == '\r') {
		line.pop_back();
	}
	if (line.size() < 2) {
		return true;
	}
	if (line[0] == 'v' && line[1] == ' ') {
		double x, y, z;
		if (std::sscanf(line.c_str() + 2, "%lf %lf %lf", &x, &y, &z) != 3) {
			return fail("base.obj line %llu: bad vertex", (unsigned long long)p.lineno);
		}
		b.v.push_back(x), b.v.push_back(y), b.v.push_back(z);
	} else if (line[0] == 'g' && line[1] == ' ') {
		p.group = line.substr(2);
	} else if (line[0] == 'f' && line[1] == ' ') {
		uint32_t f[64];
		size_t nf = 0;
		const char *s = line.c_str() + 2;
		while (*s) {
			while (*s == ' ' || *s == '\t') {
				++s;
			}
			if (!*s) {
				break;
			}
			char *q = nullptr;
			const long k = std::strtol(s, &q, 10);
			if (q == s || k < 1 || nf == 64) {
				return fail("base.obj line %llu: bad face index", (unsigned long long)p.lineno);
			}
			f[nf++] = uint32_t(k - 1);
			s = q;
			while (*s && *s != ' ' && *s != '\t') {
				++s; // `/vt/vn` suffixes
			}
		}
		if (nf < 3) {
			return fail("base.obj line %llu: face with %zu corners", (unsigned long long)p.lineno, nf);
		}
		++b.nfaces;
		if (nf == 4) {
			++b.nquads;
		} else if (nf == 3) {
			++b.ntris;
		}
		if (p.group == "body") {
			for (size_t k = 1; k + 1 < nf; ++k) {
				b.body_tris.push_back(f[0]), b.body_tris.push_back(f[k]), b.body_tris.push_back(f[k + 1]);
			}
		} else if (p.group == "joint-neck") {
			p.neck.insert(p.neck.end(), f, f + nf);
		}
	}
	return true;
}

bool base_finish(Parse &p) {
	Base &b = p.b;
	std::vector<double> &v = b.v;
	b.nv = uint32_t(v.size() / 3);
	for (uint32_t k : b.body_tris) {
		if (k >= b.nv) {
			return fail("base.obj: face index %u past %u vertices", k + 1, b.nv);
		}
	}
	if (p.neck.empty()) {
		return fail("base.obj: no joint-neck group");
	}
	std::sort(p.neck.begin(), p.neck.end());
	p.neck.erase(std::unique(p.neck.begin(), p.neck.end()), p.neck.end());
	double ys = 0.0;
	for (uint32_t k : p.neck) {
		if (k >= b.nv) {
			return fail("base.obj: joint-neck index past the vertices");
		}
		ys += v[k * 3 + 1];
	}
	b.y_neck = ys / double(p.neck.size());
	b.neck_cube_verts = uint32_t(p.neck.size());
	std::vector<uint8_t> body(b.nv, 0);
	for (uint32_t k : b.body_tris) {
		body[k] = 1;
	}
	b.cand.assign(b.nv, -1);
	for (uint32_t k = 0; k < b.nv; ++k) {
		if (body[k] && v[k * 3 + 1] >= b.y_neck - kNeckBandDm) {
			b.cand[k] = int32_t(b.cand_ids.size());
			b.cand_ids.push_back(k);
		}
	}
	b.ok = true;
	g_base = std::move(b);
	++g_base_gen;
	tfiles().clear(); // targets are indexed on it: re-upload them
	return true;
}

// `idx dx dy dz` per line (MPFB2's target format, 0-based indices on base.obj).
bool target_line(Parse &p, const char *s, const char *end) {
	TFile &t = p.t;
	while (s < end && (*s == ' ' || *s == '\t' || *s == '\r')) {
		++s;
	}
	if (s == end) {
		return true; // a blank line
	}
	char *q = nullptr;
	const long long idx = std::strtoll(s, &q, 10);
	if (q == s || idx < 0 || idx >= (long long)g_base.nv) {
		return fail("%s line %llu: bad index (the file has %u vertices to index)", p.key.c_str(),
				(unsigned long long)p.lineno, g_base.nv);
	}
	double d[3];
	const char *r = q;
	for (int k = 0; k < 3; ++k) {
		char *q2 = nullptr;
		d[k] = std::strtod(r, &q2);
		if (q2 == r || q2 > end || !std::isfinite(d[k])) {
			return fail("%s line %llu: expected `idx dx dy dz`", p.key.c_str(), (unsigned long long)p.lineno);
		}
		r = q2;
	}
	while (r < end && (*r == ' ' || *r == '\t' || *r == '\r')) {
		++r;
	}
	if (r != end) {
		return fail("%s line %llu: trailing text after `idx dx dy dz`", p.key.c_str(), (unsigned long long)p.lineno);
	}
	++t.lines;
	t.idx_sum += idx;
	t.idx_max = std::max<int64_t>(t.idx_max, idx);
	t.sum += d[0] + d[1] + d[2];
	t.abs_sum += std::fabs(d[0]) + std::fabs(d[1]) + std::fabs(d[2]);
	const int32_t c = g_base.cand[size_t(idx)];
	if (c >= 0) {
		t.ci.push_back(uint32_t(c));
		for (int k = 0; k < 3; ++k) {
			t.d.push_back(float(d[k] * kDmToMm));
		}
	} else if (std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * kDmToMm > 0.1) {
		++t.dropped_moving;
	}
	return true;
}

// One item: up to kLinesPerItem lines of the queue's head; the file's end
// finishes it. False on a parse failure (the file is dropped, g_load_err
// keeps the first).
bool parse_item(Gas &gas) {
	Parse &p = parse_queue().front();
	if (!p.base && !g_base.ok) {
		fail("%s: upload mpfb2/3dobjs/base.obj first (targets are indexed on it)", p.key.c_str());
	} else {
		bool ok = true;
		uint32_t n = 0;
		while (ok && n < kLinesPerItem && p.pos < p.text.size()) {
			size_t e = p.text.find('\n', p.pos);
			if (e == std::string::npos) {
				e = p.text.size();
			}
			++p.lineno;
			ok = p.base ? base_line(p, p.text.substr(p.pos, e - p.pos))
						: target_line(p, p.text.c_str() + p.pos, p.text.c_str() + e);
			p.pos = e + 1;
			++n;
		}
		gas.spend(16);
		if (ok && p.pos < p.text.size()) {
			return true;
		}
		if (ok && (p.base ? base_finish(p) : true)) {
			if (!p.base) {
				tfiles()[p.key] = std::move(p.t);
			}
			++g_parsed;
			parse_queue().pop_front();
			return true;
		}
	}
	if (g_load_err.empty()) {
		g_load_err = g_err;
	}
	parse_queue().pop_front();
	return false;
}

// Drain the queue under gas. "" when drained, else a status.
std::string loader_tick(Gas &gas) {
	while (!parse_queue().empty() && gas.more()) {
		parse_item(gas);
	}
	if (!parse_queue().empty()) {
		return fmt("RUNNING loader queued %zu parsed %llu", parse_queue().size(), (unsigned long long)g_parsed);
	}
	return "";
}

// ---- the phenotype anchor rule (anny-creator's scripts/anny_coeffs.gd) -------

// The 17 free slots: male female | newborn baby child young old | min avg max muscle
// | min avg max weight | min max height | ideal uncommon proportions
const char *const kAxes[6] = { "gender", "age", "muscle", "weight", "height", "proportions" };
const int kAxisFirst[6] = { 0, 2, 7, 10, 13, 15 };
const int kAxisCount[6] = { 2, 5, 3, 3, 2, 2 };
// anny_tables.json's anchors, to the bit.
const double kAnchors[6][5] = {
	{ 0.0, 1.0 },
	{ -0.3333333333333333, 0.0, 0.33333333333333337, 0.6666666666666667, 1.0 },
	{ 0.0, 0.5, 1.0 },
	{ 0.0, 0.5, 1.0 },
	{ 0.0, 1.0 },
	{ 0.0, 1.0 },
};
int slot_axis(int s) {
	for (int a = 5; a >= 0; --a) {
		if (s >= kAxisFirst[a]) {
			return a;
		}
	}
	return 0;
}

// interp(value, anchors) -> weights and their derivative d w / d value.
void interp(double value, int axis, double w[5], double dw[5]) {
	const int n = kAxisCount[axis];
	const double *an = kAnchors[axis];
	for (int k = 0; k < 5; ++k) {
		w[k] = 0.0, dw[k] = 0.0;
	}
	int idx = 1;
	while (idx < n - 1 && an[idx] < value) {
		++idx;
	}
	const double lo = an[idx - 1], hi = an[idx];
	const double raw = (value - lo) / (hi - lo);
	const double alpha = std::min(std::max(raw, 0.0), 1.0);
	w[idx - 1] = 1.0 - alpha;
	w[idx] = alpha;
	if (raw >= 0.0 && raw <= 1.0) {
		dw[idx - 1] = -1.0 / (hi - lo);
		dw[idx] = 1.0 / (hi - lo);
	}
}

// ---- the model: handles, sources, the coefficient rule --------------------------

struct Mesh {
	// mm, centred on the vertices' centroid (origin, in the mesh's own mm): the
	// fit's translation then stays near 0, where float32 resolves it (at
	// 1,141 mm its ulp is 1.2e-4 mm, and an L-BFGS-B step below it stalled
	// the solve; gates/11-headfit, the oracle).
	hfg::Mesh m;
	double origin[3] = { 0, 0, 0 };
	std::vector<std::string> shape_names;
	std::vector<std::vector<float>> shapes; // the avatar's own shapes (hf_mesh_add_shape), dense, metres
};

struct Marks {
	std::vector<int32_t> model_ids, avatar_ids;
};


enum Kind { K_MESH, K_MODEL, K_MARKS, K_FIT, K_XFER };
struct Handle {
	Kind kind;
	std::shared_ptr<void> p;
};
std::map<int, Handle> &handles() {
	static std::map<int, Handle> h;
	return h;
}
int g_next = 1;

template <class T>
T *get(int h, Kind k, const char *what) {
	auto it = handles().find(h);
	if (it == handles().end() || it->second.kind != k) {
		fail("%d is not a live %s handle", h, what);
		return nullptr;
	}
	return static_cast<T *>(it->second.p.get());
}

template <class T>
int put(Kind k, std::shared_ptr<T> p) {
	const int h = g_next++;
	handles()[h] = Handle{ k, std::static_pointer_cast<void>(p) };
	return h;
}

const TFile *need(const std::string &key) {
	auto it = tfiles().find(key);
	if (it == tfiles().end()) {
		fail("missing ANNY target %s (upload it with hf_data_put)", key.c_str());
		return nullptr;
	}
	return &it->second;
}

// Add a file's region entries, times `scale`, into row (P*3) through the
// candidate -> region map.
void add_into(float *row, const TFile &t, const std::vector<int32_t> &c2r, double scale) {
	for (size_t e = 0; e < t.ci.size(); ++e) {
		const int32_t r = c2r[t.ci[e]];
		if (r < 0) {
			continue;
		}
		for (int k = 0; k < 3; ++k) {
			row[size_t(r) * 3 + k] = float(double(row[size_t(r) * 3 + k]) + scale * double(t.d[e * 3 + k]));
		}
	}
}

const char *const kFacial[52] = { "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
	"browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
	"eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
	"eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
	"jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
	"mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
	"mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
	"mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
	"mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight", "tongueOut" };

int facial_index(const std::string &n) {
	for (int k = 0; k < 52; ++k) {
		if (n == kFacial[k]) {
			return k;
		}
	}
	return -1;
}

// local_dials.txt: `group name pos_file[,pos_file] neg_file[,neg_file]`
// (files relative to mpfb2/targets/<group>/, without .target).
struct DialSpec {
	std::string group, name;
	std::vector<std::string> pos, neg;
};
bool parse_dials(std::vector<DialSpec> &out) {
	auto it = raw().find("local_dials.txt");
	if (it == raw().end()) {
		return fail("missing local_dials.txt (upload guest/headfit/local_dials.txt with hf_data_put)");
	}
	std::istringstream is(it->second);
	std::string line;
	int ln = 0;
	auto split = [](const std::string &s) {
		std::vector<std::string> r;
		std::string cur;
		for (char c : s) {
			if (c == ',') {
				r.push_back(cur), cur.clear();
			} else {
				cur += c;
			}
		}
		r.push_back(cur);
		return r;
	};
	while (std::getline(is, line)) {
		++ln;
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::istringstream ls(line);
		DialSpec d;
		std::string p, n;
		if (!(ls >> d.group >> d.name >> p >> n)) {
			return fail("local_dials.txt line %d: expected `group name pos neg`", ln);
		}
		d.pos = split(p), d.neg = split(n);
		out.push_back(d);
	}
	return true;
}

std::string macro_key(const std::string &rel) {
	return "mpfb2/targets/macrodetails/" + rel + ".target";
}

// One macro row: its name, the files (with weights) it sums, its slots.
struct MacroRow {
	std::string name;
	std::vector<std::pair<std::string, double>> files;
	std::vector<int> slots;
	bool newborn = false;
};

std::vector<MacroRow> macro_rows() {
	std::vector<MacroRow> rows;
	const char *G[2] = { "male", "female" };
	const char *A[5] = { "newborn", "baby", "child", "young", "old" };
	const char *M[3] = { "minmuscle", "averagemuscle", "maxmuscle" };
	const char *W[3] = { "minweight", "averageweight", "maxweight" };
	const char *H[2] = { "minheight", "maxheight" };
	const char *Pp[2] = { "idealproportions", "uncommonproportions" };
	const char *R[3] = { "african", "asian", "caucasian" };
	auto file_age = [](int a) { return std::string(a == 0 ? "baby" : (a == 1 ? "baby" : (a == 2 ? "child" : (a == 3 ? "young" : "old")))); };
	// ANNY's macrodetails order (full_model.py): universal, race, height, proportions.
	for (int g = 0; g < 2; ++g)
		for (int a = 0; a < 5; ++a)
			for (int m = 0; m < 3; ++m)
				for (int w = 0; w < 3; ++w) {
					MacroRow r;
					r.name = fmt("mac_universal_%s-%s-%s-%s", G[g], A[a], M[m], W[w]);
					r.files.push_back({ macro_key(fmt("universal-%s-%s-%s-%s", G[g], file_age(a).c_str(), M[m], W[w])), 1.0 });
					r.slots = { g, 2 + a, 7 + m, 10 + w };
					r.newborn = a == 0;
					rows.push_back(r);
				}
	for (int g = 0; g < 2; ++g)
		for (int a = 0; a < 5; ++a) {
			MacroRow r;
			r.name = fmt("mac_race_%s-%s", G[g], A[a]);
			// race is not a free axis: bake_anny.py folds it at 1/3 each.
			for (int k = 0; k < 3; ++k) {
				r.files.push_back({ macro_key(fmt("%s-%s-%s", R[k], G[g], file_age(a).c_str())), 1.0 / 3.0 });
			}
			r.slots = { g, 2 + a };
			r.newborn = a == 0;
			rows.push_back(r);
		}
	for (int g = 0; g < 2; ++g)
		for (int a = 0; a < 5; ++a)
			for (int m = 0; m < 3; ++m)
				for (int w = 0; w < 3; ++w)
					for (int h = 0; h < 2; ++h) {
						MacroRow r;
						r.name = fmt("mac_height_%s-%s-%s-%s-%s", G[g], A[a], M[m], W[w], H[h]);
						r.files.push_back({ macro_key(fmt("height/%s-%s-%s-%s-%s", G[g], file_age(a).c_str(), M[m], W[w], H[h])), 1.0 });
						r.slots = { g, 2 + a, 7 + m, 10 + w, 13 + h };
						r.newborn = a == 0;
						rows.push_back(r);
					}
	for (int g = 0; g < 2; ++g)
		for (int a = 2; a < 5; ++a)
			for (int m = 0; m < 3; ++m)
				for (int w = 0; w < 3; ++w)
					for (int p = 0; p < 2; ++p) {
						MacroRow r;
						r.name = fmt("mac_proportions_%s-%s-%s-%s-%s", G[g], A[a], M[m], W[w], Pp[p]);
						r.files.push_back({ macro_key(fmt("proportions/%s-%s-%s-%s-%s", G[g], A[a], M[m], W[w], Pp[p])), 1.0 });
						r.slots = { g, 2 + a, 7 + m, 10 + w, 15 + p };
						rows.push_back(r);
					}
	return rows;
}

// ---- the model, built as a job ------------------------------------------------

struct Model {
	int region = 0;
	uint32_t P = 0;
	std::vector<uint32_t> gid; // region vertex -> base.obj index (0-based)
	std::vector<float> base; // P*3 mm, centred
	double centre[3] = { 0, 0, 0 }; // subtracted from the mm positions
	std::vector<uint32_t> tri; // region triangles
	std::vector<uint8_t> label; // per vertex: R_FACE..R_NECK
	std::vector<int32_t> c2r; // candidate -> region vertex, or -1 (unit: fields)
	uint32_t P_head = 0, P_neck = 0; // the fit region and the neck band
	uint32_t NB = 0, macro_rows = 0, L = 0;
	std::vector<float> blend; // NB*P*3
	std::vector<std::string> row_names;
	std::vector<int8_t> row_slots; // macro_rows*5, -1 padded
	std::vector<std::string> dial_names;
	std::vector<std::string> fa_names; // the 52 facial actions
	std::vector<float> fa_d; // 52*P*3 mm
	std::vector<uint32_t> fa_file_support; // entries in the whole file
	std::vector<uint32_t> fa_region_support; // non-zero entries in the region
	double y_lip = 0.0; // the lip line (mm, model frame) for mask:upper/lower
	uint32_t files_macro = 0, files_dial = 0, files_facial = 0;
	uint64_t lines_macro = 0, lines_dial = 0, lines_facial = 0;
	// The job.
	enum State { M_LOAD, M_LABELS, M_REGION, M_MACRO, M_DIALS, M_FACIAL, M_LIP, M_DONE, M_FAILED } state = M_LOAD;
	uint64_t base_gen = 0;
	uint32_t cursor = 0;
	std::string err;
	std::vector<DialSpec> dials;
	std::vector<uint8_t> clab; // per candidate
	struct MarkTask {
		std::string key;
		uint8_t lab;
		double thr_mm;
	};
	std::vector<MarkTask> marks;
	std::vector<MacroRow> rows;
	std::vector<double> tmpl; // P*3 mm, base.obj's frame
	std::map<std::string, int> seen;
	bool ready() const { return state == M_DONE; }
};

void coefficients(const Model &md, const float *phen, const float *dials, std::vector<float> &c) {
	double sv[17];
	for (int a = 0; a < 6; ++a) {
		double w[5], dw[5];
		interp(phen[a], a, w, dw);
		for (int k = 0; k < kAxisCount[a]; ++k) {
			sv[kAxisFirst[a] + k] = w[k];
		}
	}
	c.assign(md.NB, 0.0f);
	for (uint32_t t = 0; t < md.macro_rows; ++t) {
		double w = 1.0;
		for (int k = 0; k < 5; ++k) {
			const int s = md.row_slots[t * 5 + k];
			if (s >= 0) {
				w *= sv[s];
			}
		}
		c[t] = float(w);
	}
	for (uint32_t k = 0; k < 2 * md.L; ++k) {
		c[md.macro_rows + k] = dials[k];
	}
}

// d f / d phen[a] = sum_t dcoeffs[t] d c_t / d phen[a].
void phen_gradient(const Model &md, const float *phen, const float *dco, double gp[6]) {
	double sv[17], dsv[17];
	for (int a = 0; a < 6; ++a) {
		double w[5], dw[5];
		interp(phen[a], a, w, dw);
		for (int k = 0; k < kAxisCount[a]; ++k) {
			sv[kAxisFirst[a] + k] = w[k];
			dsv[kAxisFirst[a] + k] = dw[k];
		}
		gp[a] = 0.0;
	}
	for (uint32_t t = 0; t < md.macro_rows; ++t) {
		const int8_t *sl = &md.row_slots[t * 5];
		for (int k = 0; k < 5; ++k) {
			if (sl[k] < 0) {
				continue;
			}
			double prod = dsv[sl[k]];
			for (int j = 0; j < 5; ++j) {
				if (j != k && sl[j] >= 0) {
					prod *= sv[sl[j]];
				}
			}
			gp[slot_axis(sl[k])] += double(dco[t]) * prod;
		}
	}
}


// The rows a fit needs at a phenotype: a macro row whose every slot has a
// non-zero weight or a non-zero derivative (any other row has coefficient 0
// and adds exactly 0 to the phenotype gradient), and every dial row. The
// pose and the gradient over these rows equal the dense ones.
void active_rows(const Model &md, const float *phen, std::vector<uint32_t> &act) {
	double sv[17], dsv[17];
	for (int a = 0; a < 6; ++a) {
		double w[5], dw[5];
		interp(phen[a], a, w, dw);
		for (int k = 0; k < kAxisCount[a]; ++k) {
			sv[kAxisFirst[a] + k] = w[k];
			dsv[kAxisFirst[a] + k] = dw[k];
		}
	}
	act.clear();
	for (uint32_t t = 0; t < md.macro_rows; ++t) {
		bool on = true;
		for (int k = 0; k < 5 && on; ++k) {
			const int s = md.row_slots[t * 5 + k];
			on = s < 0 || sv[s] != 0.0 || dsv[s] != 0.0;
		}
		if (on) {
			act.push_back(t);
		}
	}
	for (uint32_t t = md.macro_rows; t < md.NB; ++t) {
		act.push_back(t);
	}
}

std::string model_fail(Model &md) {
	md.state = Model::M_FAILED;
	md.err = g_err;
	return "FAIL " + md.err;
}

// One model item (or the loader's, while the queue is not empty).
std::string model_item(Model &md, Gas &gas) {
	const Base &b = g_base;
	switch (md.state) {
		case Model::M_LOAD: {
			const std::string s = loader_tick(gas);
			if (!s.empty()) {
				return s;
			}
			if (!g_load_err.empty()) {
				g_err = "the upload did not parse: " + g_load_err;
				return model_fail(md);
			}
			if (!b.ok) {
				fail("hf_model_load: no base.obj (upload mpfb2/3dobjs/base.obj first)");
				return model_fail(md);
			}
			md.base_gen = g_base_gen;
			// Labels from the sources themselves: the eyelids are the support of
			// eyeBlink and eyeWide, the nose and the mouth the support of their
			// local dials (> 0.5 mm), the face the support of the 52 actions.
			// The neck band keeps its own label: no source relabels it.
			if (!parse_dials(md.dials)) {
				return model_fail(md);
			}
			const uint32_t NC = uint32_t(b.cand_ids.size());
			md.clab.assign(NC, R_CRANIUM);
			for (uint32_t c = 0; c < NC; ++c) {
				if (b.v[size_t(b.cand_ids[c]) * 3 + 1] < b.y_neck) {
					md.clab[c] = R_NECK;
				}
			}
			const std::string fbase = "faceunits01/targets/faceunits/";
			for (int k = 0; k < 52; ++k) {
				md.marks.push_back({ fbase + kFacial[k] + ".target", R_FACE, 0.1 });
			}
			for (const DialSpec &d : md.dials) {
				if (d.group != "mouth" && d.group != "nose") {
					continue;
				}
				for (const auto *lst : { &d.pos, &d.neg }) {
					for (const std::string &f : *lst) {
						md.marks.push_back(
								{ "mpfb2/targets/" + d.group + "/" + f + ".target", uint8_t(d.group == "mouth" ? R_MOUTH : R_NOSE), 0.5 });
					}
				}
			}
			for (const char *e : { "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight" }) {
				md.marks.push_back({ fbase + e + ".target", R_EYES, 0.1 });
			}
			md.state = Model::M_LABELS;
			md.cursor = 0;
			gas.spend(kilo(NC));
			return "";
		}
		case Model::M_LABELS: {
			const Model::MarkTask &mt = md.marks[md.cursor];
			const TFile *t = need(mt.key);
			if (!t) {
				return model_fail(md);
			}
			for (size_t e = 0; e < t->ci.size(); ++e) {
				const double m = std::sqrt(double(t->d[e * 3]) * t->d[e * 3] + double(t->d[e * 3 + 1]) * t->d[e * 3 + 1] +
						double(t->d[e * 3 + 2]) * t->d[e * 3 + 2]);
				if (m > mt.thr_mm && md.clab[t->ci[e]] != R_NECK) {
					md.clab[t->ci[e]] = mt.lab;
				}
			}
			gas.spend(kilo(t->ci.size() * 4));
			if (++md.cursor == md.marks.size()) {
				md.state = Model::M_REGION;
			}
			return "";
		}
		case Model::M_REGION: {
			// The region: the head and its neck band (every candidate), or the
			// face (all but the cranium and the neck).
			const uint32_t NC = uint32_t(b.cand_ids.size());
			md.c2r.assign(NC, -1);
			for (uint32_t c = 0; c < NC; ++c) {
				if (md.region == 0 || (md.clab[c] != R_CRANIUM && md.clab[c] != R_NECK)) {
					md.c2r[c] = int32_t(md.gid.size());
					md.gid.push_back(b.cand_ids[c]);
					md.label.push_back(md.clab[c]);
				}
			}
			md.P = uint32_t(md.gid.size());
			if (md.P == 0) {
				fail("hf_model_load: region %d is empty", md.region);
				return model_fail(md);
			}
			std::vector<int32_t> g2r(b.nv, -1);
			for (uint32_t r = 0; r < md.P; ++r) {
				g2r[md.gid[r]] = int32_t(r);
			}
			for (size_t k = 0; k + 2 < b.body_tris.size(); k += 3) {
				const int32_t a = g2r[b.body_tris[k]], c = g2r[b.body_tris[k + 1]], d = g2r[b.body_tris[k + 2]];
				if (a >= 0 && c >= 0 && d >= 0) {
					md.tri.push_back(uint32_t(a)), md.tri.push_back(uint32_t(c)), md.tri.push_back(uint32_t(d));
				}
			}
			// Base positions in mm, centred on the fit region's centroid (the
			// neck band left out, so the band changes no fitted number).
			md.tmpl.assign(size_t(md.P) * 3, 0.0);
			for (uint32_t r = 0; r < md.P; ++r) {
				const bool neck = md.label[r] == R_NECK;
				++(neck ? md.P_neck : md.P_head);
				for (int k = 0; k < 3; ++k) {
					md.tmpl[r * 3 + k] = b.v[size_t(md.gid[r]) * 3 + k] * kDmToMm;
					if (!neck) {
						md.centre[k] += md.tmpl[r * 3 + k];
					}
				}
			}
			for (int k = 0; k < 3; ++k) {
				md.centre[k] /= double(md.P_head);
			}
			md.base.resize(size_t(md.P) * 3);
			for (uint32_t r = 0; r < md.P; ++r) {
				for (int k = 0; k < 3; ++k) {
					md.base[r * 3 + k] = float(md.tmpl[r * 3 + k] - md.centre[k]);
				}
			}
			md.rows = macro_rows();
			md.macro_rows = uint32_t(md.rows.size());
			md.L = uint32_t(md.dials.size());
			md.NB = md.macro_rows + 2 * md.L;
			md.blend.assign(size_t(md.NB) * md.P * 3, 0.0f);
			md.row_names.clear();
			md.row_slots.clear();
			md.state = Model::M_MACRO;
			md.cursor = 0;
			gas.spend(kilo(size_t(md.NB) * md.P * 3 + b.body_tris.size()));
			return "";
		}
		case Model::M_MACRO: {
			// One macro row. Newborn rows scale the baby files by (0.922, 0.922,
			// 0.75) in ANNY's Z-up world, which is (0.922, 0.75, 0.922) on
			// base.obj's Y-up axes, plus (scale - 1)/3 of the template
			// (full_model.py).
			const MacroRow &r = md.rows[md.cursor];
			float *row = &md.blend[size_t(md.cursor) * md.P * 3];
			uint64_t work = md.P;
			for (const auto &f : r.files) {
				const TFile *t = need(f.first);
				if (!t) {
					return model_fail(md);
				}
				if (!md.seen.count(f.first)) {
					md.seen[f.first] = 1;
					++md.files_macro;
					md.lines_macro += t->lines;
				}
				add_into(row, *t, md.c2r, f.second);
				work += t->ci.size();
			}
			if (r.newborn) {
				const double nbS[3] = { 0.922, 0.75, 0.922 };
				for (uint32_t i = 0; i < md.P; ++i) {
					for (int k = 0; k < 3; ++k) {
						const double v = nbS[k] * double(row[i * 3 + k]) + (nbS[k] - 1.0) / 3.0 * md.tmpl[i * 3 + k];
						row[i * 3 + k] = float(v);
					}
				}
			}
			md.row_names.push_back(r.name);
			for (int k = 0; k < 5; ++k) {
				md.row_slots.push_back(int8_t(k < int(r.slots.size()) ? r.slots[k] : -1));
			}
			gas.spend(kilo(work * 3));
			if (++md.cursor == md.macro_rows) {
				md.state = Model::M_DIALS;
				md.cursor = 0;
			}
			return "";
		}
		case Model::M_DIALS: {
			// A local dial: a positive and a negative row (both sides of a sided
			// dial move together: the face stays symmetric).
			const DialSpec &d = md.dials[md.cursor];
			uint64_t work = 0;
			for (int side = 0; side < 2; ++side) {
				const std::vector<std::string> &lst = side == 0 ? d.pos : d.neg;
				float *row = &md.blend[size_t(md.macro_rows + 2 * md.cursor + side) * md.P * 3];
				for (const std::string &f : lst) {
					const TFile *t = need("mpfb2/targets/" + d.group + "/" + f + ".target");
					if (!t) {
						return model_fail(md);
					}
					++md.files_dial;
					md.lines_dial += t->lines;
					add_into(row, *t, md.c2r, 1.0);
					work += t->ci.size();
				}
				md.row_names.push_back(d.name + (side == 0 ? "+" : "-"));
			}
			md.dial_names.push_back(d.name);
			gas.spend(kilo(work * 3));
			if (++md.cursor == md.L) {
				md.state = Model::M_FACIAL;
				md.cursor = 0;
				md.fa_d.assign(size_t(52) * md.P * 3, 0.0f);
			}
			return "";
		}
		case Model::M_FACIAL: {
			const int k = int(md.cursor);
			const TFile *t = need(std::string("faceunits01/targets/faceunits/") + kFacial[k] + ".target");
			if (!t) {
				return model_fail(md);
			}
			++md.files_facial;
			md.lines_facial += t->lines;
			md.fa_names.push_back(kFacial[k]);
			md.fa_file_support.push_back(uint32_t(t->lines));
			float *row = &md.fa_d[size_t(k) * md.P * 3];
			add_into(row, *t, md.c2r, 1.0);
			uint32_t nz = 0;
			for (uint32_t i = 0; i < md.P; ++i) {
				if (row[i * 3] != 0.0f || row[i * 3 + 1] != 0.0f || row[i * 3 + 2] != 0.0f) {
					++nz;
				}
			}
			md.fa_region_support.push_back(nz);
			gas.spend(kilo((t->ci.size() + md.P) * 3));
			if (++md.cursor == 52) {
				md.state = Model::M_LIP;
			}
			return "";
		}
		case Model::M_LIP: {
			// The lip line: halfway between the mean heights of mouthRollUpper's
			// and mouthRollLower's support near the midline (|x| < 10 mm).
			auto mean_y = [&](int fa) {
				double s = 0.0;
				int n = 0;
				for (uint32_t i = 0; i < md.P; ++i) {
					const float *d = &md.fa_d[(size_t(fa) * md.P + i) * 3];
					if ((d[0] != 0.0f || d[1] != 0.0f || d[2] != 0.0f) && std::fabs(md.base[i * 3]) < 10.0f) {
						s += md.base[i * 3 + 1];
						++n;
					}
				}
				return n ? s / n : 0.0;
			};
			md.y_lip = 0.5 * (mean_y(facial_index("mouthRollUpper")) + mean_y(facial_index("mouthRollLower")));
			md.state = Model::M_DONE;
			md.clab.clear(), md.clab.shrink_to_fit();
			md.marks.clear();
			gas.spend(kilo(md.P * 2));
			return "";
		}
		case Model::M_DONE:
			return fmt("DONE model region %d P %u NB %u", md.region, md.P, md.NB);
		default:
			return "FAIL " + md.err;
	}
}

std::string model_tick(Model &md, Gas &gas) {
	while (gas.more()) {
		const std::string s = model_item(md, gas);
		if (!s.empty()) {
			return s;
		}
	}
	return md.state == Model::M_DONE ? fmt("DONE model region %d P %u NB %u", md.region, md.P, md.NB)
									 : fmt("RUNNING model state %d item %u", int(md.state), md.cursor);
}

// ---- the unified-expression map -------------------------------------------------
//
// unified_expressions.map (RFD 2275's catalog), one name per line:
//   Name <= src [* mask:m]... [+ src [* mask:m]...]...   `*` binds tighter than `+`
// (a factor is mask:left|right|upper|lower or a number, `jawForward * -1`)
// where src is one of ANNY's 52 actions, another name of the map (a sum of
// entries), artist:<avatar shape>, procedural:<kind>, corrective:A*B (A and B
// names of the map), unit:<name> (MPFB2's expression unit, race-folded),
// bone:<name> or baked:<skin> (RFD 2279), or
// material:<property> (a material parameter driven instead of a shape); or
//   Name <= absent: <reason>
// `# ...` is a comment and `@stage N` starts a section.

// unit:<name> is MPFB2's expression unit, one file per race, folded at 1/3
// each as the macro rows fold race (bake_anny.py; race is not a free axis).
const char *const kRaces[3] = { "african", "asian", "caucasian" };
constexpr double kRaceFold = 1.0 / 3.0;
std::string unit_key(const char *race, const std::string &name) {
	return std::string("mpfb2/targets/expression/units/") + race + "/" + name + ".target";
}

const char *const kProcedural[7] ={ "out_step1", "out_step2", "roll", "twist_left", "twist_right", "flat", "squish" };

struct Term {
	enum K { ACTION, ENTRY, UNIT, ARTIST, PROCEDURAL, CORRECTIVE, BONE, BAKED, MATERIAL, ABSENT } k = ACTION;
	std::string ref; // the action, entry, shape, kind, pair, bone, skin or reason
	std::vector<std::string> masks; // left, right, upper, lower; applied in order
	double scale = 1.0; // `* <number>`: a linear source at that weight
	std::string a, b; // a corrective's operands
};
struct Entry {
	std::string name;
	std::vector<Term> terms; // a sum
	std::string form; // direct, mask, artist, sum, procedural, corrective, bone, baked, material, absent
	int line = 0;
	int stage = 0;
};

std::string trim(const std::string &s) {
	size_t a = 0, b = s.size();
	while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) {
		++a;
	}
	while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) {
		--b;
	}
	return s.substr(a, b - a);
}

std::vector<std::string> split_on(const std::string &s, char c) {
	std::vector<std::string> r;
	std::string cur;
	for (char ch : s) {
		if (ch == c) {
			r.push_back(trim(cur)), cur.clear();
		} else {
			cur += ch;
		}
	}
	r.push_back(trim(cur));
	return r;
}

bool has_prefix(const std::string &s, const char *p, std::string &rest) {
	const size_t n = std::strlen(p);
	if (s.compare(0, n, p) != 0) {
		return false;
	}
	rest = trim(s.substr(n));
	return true;
}

// Parse the map's syntax; names are resolved by resolve_map. Syntax errors go
// to errs with their line; false only when the map was never uploaded.
bool parse_map(std::vector<Entry> &out, std::string &errs) {
	auto it = raw().find("unified_expressions.map");
	if (it == raw().end()) {
		return fail("missing unified_expressions.map (upload it with hf_data_put)");
	}
	std::istringstream is(it->second);
	std::string line;
	int ln = 0, stage = 0;
	while (std::getline(is, line)) {
		++ln;
		const size_t hash = line.find('#');
		if (hash != std::string::npos) {
			line = line.substr(0, hash);
		}
		line = trim(line);
		if (line.empty()) {
			continue;
		}
		std::string rest;
		if (has_prefix(line, "@stage", rest)) {
			char *q = nullptr;
			const long s = std::strtol(rest.c_str(), &q, 10);
			if (rest.empty() || *q != '\0' || s < 1) {
				errs += fmt("line %d: `@stage %s` is not a stage number\n", ln, rest.c_str());
			} else {
				stage = int(s);
			}
			continue;
		}
		const size_t arrow = line.find("<=");
		if (arrow == std::string::npos) {
			errs += fmt("line %d: no `<=`\n", ln);
			continue;
		}
		Entry e;
		e.line = ln;
		e.stage = stage;
		e.name = trim(line.substr(0, arrow));
		const std::string rhs = trim(line.substr(arrow + 2));
		if (e.name.empty() || e.name.find_first_of(" \t*+:") != std::string::npos) {
			errs += fmt("line %d: bad name `%s`\n", ln, e.name.c_str());
			continue;
		}
		if (has_prefix(rhs, "absent:", rest)) {
			Term t;
			t.k = Term::ABSENT;
			t.ref = rest;
			if (rest.empty()) {
				errs += fmt("line %d: %s is absent with no reason\n", ln, e.name.c_str());
				continue;
			}
			e.terms.push_back(t);
			e.form = "absent";
			out.push_back(e);
			continue;
		}
		bool ok = true;
		for (const std::string &p : split_on(rhs, '+')) {
			Term t;
			if (p.empty()) {
				errs += fmt("line %d: an empty term\n", ln);
				ok = false;
				break;
			}
			if (has_prefix(p, "corrective:", rest)) {
				const std::vector<std::string> ab = split_on(rest, '*');
				if (ab.size() != 2 || ab[0].empty() || ab[1].empty()) {
					errs += fmt("line %d: corrective:%s is not a pair A*B\n", ln, rest.c_str());
					ok = false;
					break;
				}
				t.k = Term::CORRECTIVE, t.ref = rest, t.a = ab[0], t.b = ab[1];
				e.terms.push_back(t);
				continue;
			}
			const std::vector<std::string> f = split_on(p, '*');
			const std::string &src = f[0];
			for (size_t k = 1; k < f.size(); ++k) {
				std::string m;
				if (has_prefix(f[k], "mask:", m)) {
					if (m != "left" && m != "right" && m != "upper" && m != "lower") {
						errs += fmt("line %d: `%s` is not mask:left|right|upper|lower\n", ln, f[k].c_str());
						ok = false;
					}
					t.masks.push_back(m);
					continue;
				}
				char *q = nullptr;
				const double s = std::strtod(f[k].c_str(), &q);
				if (f[k].empty() || *q != '\0' || !std::isfinite(s) || s == 0.0) {
					errs += fmt("line %d: `%s` is neither mask:<side> nor a non-zero number\n", ln, f[k].c_str());
					ok = false;
				}
				t.scale *= s;
			}
			if (has_prefix(src, "artist:", rest)) {
				t.k = Term::ARTIST;
			} else if (has_prefix(src, "unit:", rest)) {
				t.k = Term::UNIT;
			} else if (has_prefix(src, "procedural:", rest)) {
				t.k = Term::PROCEDURAL;
			} else if (has_prefix(src, "bone:", rest)) {
				t.k = Term::BONE;
			} else if (has_prefix(src, "baked:", rest)) {
				t.k = Term::BAKED;
			} else if (has_prefix(src, "material:", rest)) {
				t.k = Term::MATERIAL;
			} else {
				t.k = Term::ACTION; // or ENTRY: resolve_map decides
				rest = src;
			}
			t.ref = rest;
			if (t.ref.empty() || t.ref.find_first_of(" \t:") != std::string::npos) {
				errs += fmt("line %d: bad source `%s`\n", ln, src.c_str());
				ok = false;
			}
			if (!t.masks.empty() && t.k != Term::ACTION && t.k != Term::UNIT) {
				errs += fmt("line %d: a mask on `%s` (masks apply to ANNY's model space only)\n", ln, src.c_str());
				ok = false;
			}
			if (t.scale != 1.0 && t.k != Term::ACTION && t.k != Term::UNIT && t.k != Term::ARTIST) {
				errs += fmt("line %d: a weight on `%s` (only linear sources take one)\n", ln, src.c_str());
				ok = false;
			}
			e.terms.push_back(t);
		}
		if (!ok) {
			continue;
		}
		const Term &t0 = e.terms[0];
		if (e.terms.size() > 1) {
			e.form = "sum";
		} else if (t0.k == Term::ARTIST) {
			e.form = "artist";
		} else if (t0.k == Term::UNIT) {
			e.form = "unit";
		} else if (t0.k == Term::PROCEDURAL) {
			e.form = "procedural";
		} else if (t0.k == Term::CORRECTIVE) {
			e.form = "corrective";
		} else if (t0.k == Term::BONE) {
			e.form = "bone";
		} else if (t0.k == Term::BAKED) {
			e.form = "baked";
		} else if (t0.k == Term::MATERIAL) {
			e.form = "material";
		} else {
			e.form = !t0.masks.empty() ? "mask" : (t0.scale != 1.0 ? "scaled" : "direct");
		}
		out.push_back(e);
	}
	return true;
}

// Name resolution: a plain source is an ANNY action, else a name of the map;
// corrective operands are names; procedural kinds are the seven RFD 2275
// derives. Each unknown source is one line of `unknown`. A sum that reaches
// itself is a cycle and is reported too.
void resolve_map(std::vector<Entry> &map, std::string &unknown, uint32_t &nunknown) {
	std::map<std::string, size_t> by;
	for (size_t k = 0; k < map.size(); ++k) {
		by[map[k].name] = k;
	}
	nunknown = 0;
	for (Entry &e : map) {
		for (Term &t : e.terms) {
			if (t.k == Term::ACTION && facial_index(t.ref) < 0) {
				if (by.count(t.ref)) {
					t.k = Term::ENTRY;
				} else {
					unknown += fmt("line %d %s: `%s` is neither one of ANNY's 52 actions nor a name of the map\n", e.line,
							e.name.c_str(), t.ref.c_str());
					++nunknown;
				}
			} else if (t.k == Term::UNIT) {
				for (const char *race : kRaces) {
					if (!tfiles().count(unit_key(race, t.ref))) {
						unknown += fmt("line %d %s: unit:%s has no %s file (upload %s)\n", e.line, e.name.c_str(),
								t.ref.c_str(), race, unit_key(race, t.ref).c_str());
						++nunknown;
					}
				}
			} else if (t.k == Term::PROCEDURAL) {
				bool known = false;
				for (const char *p : kProcedural) {
					known = known || t.ref == p;
				}
				if (!known) {
					unknown += fmt("line %d %s: procedural:%s is not a kind RFD 2275 derives\n", e.line, e.name.c_str(),
							t.ref.c_str());
					++nunknown;
				}
			} else if (t.k == Term::CORRECTIVE) {
				for (const std::string *o : { &t.a, &t.b }) {
					if (!by.count(*o)) {
						unknown += fmt("line %d %s: corrective operand `%s` is not a name of the map\n", e.line,
								e.name.c_str(), o->c_str());
						++nunknown;
					}
				}
			}
		}
	}
	// Cycles through ENTRY references (depth-first, three colours).
	std::vector<int> colour(map.size(), 0);
	std::function<bool(size_t)> visit = [&](size_t k) {
		if (colour[k] == 1) {
			return true;
		}
		if (colour[k] == 2) {
			return false;
		}
		colour[k] = 1;
		bool cyc = false;
		for (const Term &t : map[k].terms) {
			if (t.k == Term::ENTRY && visit(by[t.ref])) {
				cyc = true;
			}
		}
		colour[k] = 2;
		return cyc;
	};
	for (size_t k = 0; k < map.size(); ++k) {
		if (colour[k] == 0 && visit(k)) {
			unknown += fmt("line %d %s: its sum reaches itself\n", map[k].line, map[k].name.c_str());
			++nunknown;
		}
	}
}

// ---- the fit ------------------------------------------------------------------------

// A smooth step across [-1, 1]: 0 below, 1 above, 3t^2 - 2t^3 between.
double smooth01(double u) {
	const double t = std::min(std::max(0.5 * (u + 1.0), 0.0), 1.0);
	return t * t * (3.0 - 2.0 * t);
}


// ---- the fit ------------------------------------------------------------------------

struct Fit;

// One pose, or one evaluation of the loss and its gradient, at x, in items:
// the kernels dispatched one thread group (64) at a time over the active rows
// (active_rows). Each kernel thread writes only its own outputs, so the
// slicing changes no arithmetic; the two serial df32 kernels are one item each.
//
// An evaluation runs on the round's support: the vertices with a surface
// weight and the marked ones (Fit::S). Every other vertex has weight 0, so
// its residual and its gradient are exact zeros, and an exact zero adds
// nothing to a df32 sum: the support gives the whole region's bits at a
// quarter of the work. A pose (the correspondence search, the reports, the
// transfer) runs on every vertex.
struct Pipe {
	enum Phase { IDLE, COEF, COMPACT, BLEND, SIM, SURF, MARKPRIOR, SUMS, SIMBP, SIMBV, BLENDBW, GRAD, READY } ph = IDLE;
	bool grad = false, sub = false;
	uint32_t cur = 0;
	uint32_t PE = 0; // the vertices this pass computes (the region, or the support)
	std::vector<float> x;
	float prm[10]; // the kernels' similarity: x's, the translation in mm
	float phen[6]; // the phenotype (x[10..15] * kPScale)
	std::vector<float> coeffs, v, pos, dpos, dv, es, ek, ep, g, dprm, dco;
	double f = 0.0;
	// The compacted rows (act) over the pass's vertices, keyed by the rows,
	// the vertex set and the round that made the support.
	std::vector<uint32_t> act, want;
	std::vector<float> ablend, sbase, acoeffs, adco;
	bool compact_ok = false, compact_sub = false;
	uint64_t compact_gen = 0;
};

struct Fit {
	std::shared_ptr<Model> md;
	std::shared_ptr<Mesh> ms;
	std::shared_ptr<Marks> mk;
	float W[W_COUNT];
	std::vector<uint8_t> tbound;
	std::vector<hfg::V3> tnorm; // target vertex normals
	hfg::Bvh tbvh;
	uint32_t n = 0, K = 0;
	std::vector<float> x, lb, ub, px0, plam;
	std::vector<uint32_t> mid;
	std::vector<float> tgtK, lw;
	std::vector<float> q, nrm, w; // this round's correspondences and weights
	std::vector<double> wi;
	double wsum = 0.0;
	std::vector<hfg::V3> mn; // the model's normals at this round's pose
	// The round's support (vertices with a surface weight, and the marked
	// ones), and its correspondences, weights and marks in support order.
	std::vector<uint32_t> S, midS;
	std::vector<float> qS, nrmS, wS;
	uint64_t sub_gen = 0;
	Pipe pipe; // the solver's
	VecCpu vec;
	Lbfgsb drv{ vec };
	Lbfgsb::Status st = Lbfgsb::FAIL;
	enum State { S_BVH, S_GEOM, S_INIT_POSE, S_INIT_SIM, S_ROUND_POSE, S_ROUND_NORMALS, S_ROUND_CORR, S_ROUND_BEGIN,
		S_RUNNING, S_EVAL, S_DONE, S_FAILED } state = S_BVH;
	uint32_t cur = 0;
	int round = 0, rounds = 0;
	int evals = 0, accepts = 0, stalls = 0;
	std::vector<float> xacc; // this round's last accepted iterate, and its f
	double facc = 0.0;
	uint32_t valid = 0; // correspondences kept this round
	std::string err;
	std::vector<std::vector<float>> round_x;
	std::vector<double> round_f;
	std::vector<int> round_iters;
	std::vector<RoundProblem> round_prob; // kept for the oracle: the weights and closest points
	double f = 0.0;
	// The final pose and the transfer's hits, made once, when a transfer asks.
	Pipe rp;
	enum HState { H_NONE, H_POSE, H_NORMALS, H_BVH, H_RAYS, H_READY } hst = H_NONE;
	std::vector<float> fpos;
	std::vector<hfg::V3> fmn;
	hfg::Mesh pmesh; // the posed model (pbvh points into it)
	hfg::Bvh pbvh;
	std::vector<uint32_t> hit;
	std::vector<float> bary;
	uint32_t matched = 0;
};

void pipe_start(Pipe &p, const Model &md, const std::vector<float> &x, bool grad, uint32_t n, uint32_t K, uint32_t PS = 0) {
	p.x = x;
	p.grad = grad;
	prm_of(x.data(), p.prm);
	phen_of(x.data(), p.phen);
	p.sub = grad;
	p.PE = grad ? PS : md.P;
	p.ph = Pipe::COEF;
	p.cur = 0;
	p.coeffs.resize(md.NB);
	p.v.resize(size_t(p.PE) * 3);
	p.pos.resize(size_t(p.PE) * 3);
	if (grad) {
		p.dpos.resize(size_t(p.PE) * 3);
		p.dv.resize(size_t(p.PE) * 3);
		p.es.resize(p.PE);
		p.ek.resize(K);
		p.ep.resize(n);
		p.g.resize(n);
		p.dprm.resize(10);
		p.dco.resize(md.NB);
	}
}

// One item; true when the pose (or the evaluation) is ready.
bool pipe_item(Pipe &p, const Fit &F, Gas &gas) {
	const Model &md = *F.md;
	const uint32_t P = p.PE, G = (P + 63) / 64;
	const uint32_t NA = uint32_t(p.act.size());
	switch (p.ph) {
		case Pipe::COEF: {
			coefficients(md, p.phen, &p.x[16], p.coeffs);
			active_rows(md, p.phen, p.want);
			if (!p.compact_ok || p.want != p.act || p.compact_sub != p.sub || (p.sub && p.compact_gen != F.sub_gen)) {
				p.act = p.want;
				p.ablend.resize(p.act.size() * size_t(P) * 3);
				p.compact_ok = false;
				p.compact_sub = p.sub;
				p.compact_gen = F.sub_gen;
				if (p.sub) {
					p.sbase.resize(size_t(P) * 3);
					for (uint32_t j = 0; j < P; ++j) {
						for (int k = 0; k < 3; ++k) {
							p.sbase[j * 3 + k] = md.base[size_t(F.S[j]) * 3 + k];
						}
					}
				}
				p.ph = Pipe::COMPACT;
			} else {
				p.ph = Pipe::BLEND;
			}
			p.acoeffs.resize(p.act.size());
			for (size_t k = 0; k < p.act.size(); ++k) {
				p.acoeffs[k] = p.coeffs[p.act[k]];
			}
			p.cur = 0;
			gas.spend(kilo(size_t(md.NB) * 8));
			return false;
		}
		case Pipe::COMPACT: {
			const float *src = &md.blend[size_t(p.act[p.cur]) * md.P * 3];
			float *dst = &p.ablend[size_t(p.cur) * P * 3];
			if (p.sub) {
				for (uint32_t j = 0; j < P; ++j) {
					for (int k = 0; k < 3; ++k) {
						dst[j * 3 + k] = src[size_t(F.S[j]) * 3 + k];
					}
				}
			} else {
				std::copy(src, src + size_t(P) * 3, dst);
			}
			gas.spend(kilo(size_t(P) * 3));
			if (++p.cur == NA) {
				p.compact_ok = true;
				p.ph = Pipe::BLEND;
				p.cur = 0;
			}
			return false;
		}
		case Pipe::BLEND:
			hfk::blend(P, NA, p.sub ? p.sbase.data() : md.base.data(), p.ablend.data(), p.acoeffs.data(), p.v.data(),
					p.cur, p.cur + 1);
			gas.spend(kilo(size_t(64) * NA * 3));
			if (++p.cur == G) {
				p.ph = Pipe::SIM, p.cur = 0;
			}
			return false;
		case Pipe::SIM:
			hfk::similarity(P, p.prm, p.v.data(), p.pos.data(), p.cur, p.cur + 1);
			gas.spend(1);
			if (++p.cur == G) {
				p.cur = 0;
				p.ph = p.grad ? Pipe::SURF : Pipe::READY;
				return !p.grad;
			}
			return false;
		case Pipe::SURF:
			hfk::surface_residual(P, p.pos.data(), F.qS.data(), F.nrmS.data(), F.wS.data(), p.es.data(), p.dpos.data(),
					p.cur, p.cur + 1);
			gas.spend(1);
			if (++p.cur == G) {
				p.ph = Pipe::MARKPRIOR, p.cur = 0;
			}
			return false;
		case Pipe::MARKPRIOR:
			hfk::landmark_residual(F.K, p.pos.data(), F.midS.data(), F.tgtK.data(), F.lw.data(), p.ek.data(), p.dpos.data());
			hfk::prior(F.n, p.x.data(), F.px0.data(), F.plam.data(), p.ep.data(), p.g.data());
			gas.spend(1);
			p.ph = Pipe::SUMS;
			return false;
		case Pipe::SUMS:
			p.f = hfk::energy_sum(P, p.es.data()) + hfk::energy_sum(F.K, p.ek.data()) + hfk::energy_sum(F.n, p.ep.data());
			gas.spend(kilo(size_t(P) * 8));
			p.ph = Pipe::SIMBP;
			return false;
		case Pipe::SIMBP:
			hfk::similarity_backward_prm(P, p.prm, p.v.data(), p.dpos.data(), p.dprm.data());
			gas.spend(kilo(size_t(P) * 32));
			p.ph = Pipe::SIMBV;
			return false;
		case Pipe::SIMBV:
			hfk::similarity_backward_v(P, p.prm, p.dpos.data(), p.dv.data(), p.cur, p.cur + 1);
			gas.spend(1);
			if (++p.cur == G) {
				p.ph = Pipe::BLENDBW, p.cur = 0;
				p.adco.assign(NA, 0.0f);
			}
			return false;
		case Pipe::BLENDBW:
			hfk::blend_backward(P, NA, p.ablend.data(), p.dv.data(), p.adco.data(), p.cur, p.cur + 1);
			gas.spend(kilo(size_t(64) * P * 3 * 4));
			if (++p.cur == (NA + 63) / 64) {
				p.ph = Pipe::GRAD, p.cur = 0;
			}
			return false;
		case Pipe::GRAD: {
			std::fill(p.dco.begin(), p.dco.end(), 0.0f);
			for (size_t k = 0; k < p.act.size(); ++k) {
				p.dco[p.act[k]] = p.adco[k];
			}
			for (int k = 0; k < 10; ++k) {
				p.g[k] = float(double(p.g[k]) + double(p.dprm[k]) * (k >= 6 && k < 9 ? double(kTScale) : 1.0));
			}
			double gp[6];
			phen_gradient(md, p.phen, p.dco.data(), gp);
			for (int a = 0; a < 6; ++a) {
				p.g[10 + a] = float(double(p.g[10 + a]) + gp[a] * double(kPScale));
			}
			for (uint32_t k = 0; k < 2 * md.L; ++k) {
				p.g[16 + k] = float(double(p.g[16 + k]) + double(p.dco[md.macro_rows + k]));
			}
			gas.spend(kilo(size_t(md.macro_rows) * 25));
			p.ph = Pipe::READY;
			return true;
		}
		case Pipe::READY:
			return true;
		default:
			return false;
	}
}

// Run a pipe to the end (reports: no gas).
void pipe_run(Pipe &p, const Fit &F) {
	Gas gas{ std::numeric_limits<int64_t>::max() };
	while (!pipe_item(p, F, gas)) {
	}
}

std::vector<hfg::V3> model_normals(const Model &md, const std::vector<float> &pos) {
	hfg::Mesh m;
	m.v.resize(md.P);
	for (uint32_t i = 0; i < md.P; ++i) {
		m.v[i] = hfg::V3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
	}
	m.t = md.tri;
	return m.vertex_normals();
}

// Is the closest feature on the target's open boundary?
bool on_boundary(const Fit &F, const hfg::Bvh::Closest &c) {
	if (c.feature == 0) {
		return false;
	}
	const uint8_t *b = &F.tbound[size_t(c.tri) * 3];
	if (c.feature <= 3) {
		return b[c.feature - 1] != 0;
	}
	const int vtx = c.feature - 4; // vertex a/b/c: edges (ca, ab), (ab, bc), (bc, ca)
	return b[(vtx + 2) % 3] != 0 || b[vtx] != 0;
}

struct Corr {
	bool ok = false;
	hfg::Bvh::Closest c;
};

Corr correspond(const Fit &F, const std::vector<float> &pos, uint32_t i, const std::vector<hfg::V3> &mn, double maxdist) {
	Corr r;
	const hfg::V3 p(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
	r.c = F.tbvh.closest(p);
	if (r.c.tri < 0 || on_boundary(F, r.c) || std::sqrt(r.c.d2) > maxdist) {
		return r;
	}
	const hfg::V3 tn = F.ms->m.tri_normal(uint32_t(r.c.tri));
	if (hfg::dot(tn, mn[i]) < 0.5) {
		return r; // the normal test: a surface facing away
	}
	r.ok = true;
	return r;
}

void snapshot_problem(Fit &F) {
	const Model &md = *F.md;
	RoundProblem p;
	p.P = md.P, p.NB = md.NB, p.K = F.K, p.n = F.n, p.L = md.L;
	p.q = F.q, p.nrm = F.nrm, p.w = F.w;
	p.mid = F.mid, p.tgt = F.tgtK, p.lw = F.lw;
	p.x0 = F.x, p.lb = F.lb, p.ub = F.ub, p.prior_x0 = F.px0, p.prior_lam = F.plam;
	p.macro_rows = md.macro_rows;
	F.round_prob.push_back(std::move(p));
}

LbfgsbParams lbfgsb_params(const Fit &F) {
	LbfgsbParams p;
	p.m = 6;
	p.epsilon = 1e-4;
	p.epsilon_rel = 0.0;
	p.past = 1;
	p.delta = 1e-8;
	p.max_iterations = std::max(1, int(F.W[W_ITERS]));
	p.max_linesearch = 30;
	return p;
}

std::string fit_fail(Fit &F, const std::string &why) {
	F.state = Fit::S_FAILED;
	F.err = why;
	return "FAIL " + why;
}

// A round ends: record it, then the next round's pose, or DONE.
std::string round_end(Fit &F, double f, const std::string &reason) {
	F.round_x.push_back(F.x);
	F.round_f.push_back(f);
	F.round_iters.push_back(F.drv.iterations());
	const std::string s = fmt("round %d done f %.9g iters %d reason %s valid %u", F.round, f, F.drv.iterations(),
			reason.c_str(), F.valid);
	F.f = f;
	if (++F.round >= F.rounds) {
		F.state = Fit::S_DONE;
		return "DONE " + s;
	}
	pipe_start(F.pipe, *F.md, F.x, false, F.n, F.K);
	F.state = Fit::S_ROUND_POSE;
	return "RUNNING " + s;
}

// One fit item. A non-empty return ends the tick: a round's start or end, an
// accepted iterate, DONE or FAIL.
std::string fit_item(Fit &F, Gas &gas) {
	const Model &md = *F.md;
	switch (F.state) {
		case Fit::S_BVH:
			F.tbvh.build(F.ms->m);
			gas.spend(kilo(F.ms->m.nt() * 24));
			F.state = Fit::S_GEOM;
			return "";
		case Fit::S_GEOM: {
			F.tbound = F.ms->m.boundary_edges();
			F.tnorm = F.ms->m.vertex_normals();
			gas.spend(kilo(F.ms->m.nt() * 16));
			// The start: the landmarks' similarity from the default head (identity,
			// the default phenotype) to the marks.
			std::vector<float> x0 = F.x;
			x0[0] = 1, x0[1] = 0, x0[2] = 0, x0[3] = 0, x0[4] = 1, x0[5] = 0;
			x0[6] = x0[7] = x0[8] = 0, x0[9] = 1;
			pipe_start(F.pipe, md, x0, false, F.n, F.K);
			F.state = Fit::S_INIT_POSE;
			return "";
		}
		case Fit::S_INIT_POSE:
			if (pipe_item(F.pipe, F, gas)) {
				F.state = Fit::S_INIT_SIM;
			}
			return "";
		case Fit::S_INIT_SIM: {
			// Umeyama with the org's rotation fitter (guest/drape/similarity.h).
			std::vector<float> src, dst;
			for (uint32_t k = 0; k < F.K; ++k) {
				for (int c = 0; c < 3; ++c) {
					src.push_back(F.pipe.v[F.mid[k] * 3 + c]);
					dst.push_back(F.tgtK[k * 3 + c]);
				}
			}
			Similarity S;
			if (!similarity_fit(src.data(), dst.data(), F.K, S)) {
				return fit_fail(F, "the landmarks do not place a similarity");
			}
			// pos = s R (v - cs) + ct = s R v + (ct - s R cs)
			for (int k = 0; k < 6; ++k) {
				F.x[k] = float(S.R[k]);
			}
			for (int r = 0; r < 3; ++r) {
				const double rc = S.R[r * 3] * S.cs.x + S.R[r * 3 + 1] * S.cs.y + S.R[r * 3 + 2] * S.cs.z;
				F.x[6 + r] = float((S.ct[r] - S.s * rc) / double(kTScale));
			}
			F.x[9] = float(S.s);
			gas.spend(1);
			pipe_start(F.pipe, md, F.x, false, F.n, F.K);
			F.state = Fit::S_ROUND_POSE;
			return "";
		}
		case Fit::S_ROUND_POSE:
			if (pipe_item(F.pipe, F, gas)) {
				F.state = Fit::S_ROUND_NORMALS;
			}
			return "";
		case Fit::S_ROUND_NORMALS:
			F.mn = model_normals(md, F.pipe.pos);
			F.wi.assign(md.P, 0.0);
			F.wsum = 0.0;
			F.valid = 0;
			F.cur = 0;
			gas.spend(kilo(size_t(md.tri.size()) * 8));
			F.state = Fit::S_ROUND_CORR;
			return "";
		case Fit::S_ROUND_CORR: {
			// 64 model vertices: each one's closest avatar point, kept or not.
			const float wr[R_COUNT] = { 0.0f, F.W[W_FACE], F.W[W_MOUTH], F.W[W_NOSE], F.W[W_EYES], F.W[W_CRANIUM], 0.0f };
			const uint32_t end = std::min(md.P, F.cur + 64);
			for (uint32_t i = F.cur; i < end; ++i) {
				const Corr c = correspond(F, F.pipe.pos, i, F.mn, F.W[W_MAXDIST]);
				const hfg::V3 tn = c.c.tri >= 0 ? F.ms->m.tri_normal(uint32_t(c.c.tri)) : hfg::V3();
				F.q[i * 3] = float(c.c.q.x), F.q[i * 3 + 1] = float(c.c.q.y), F.q[i * 3 + 2] = float(c.c.q.z);
				F.nrm[i * 3] = float(tn.x), F.nrm[i * 3 + 1] = float(tn.y), F.nrm[i * 3 + 2] = float(tn.z);
				if (c.ok && wr[md.label[i]] > 0.0f) {
					F.wi[i] = wr[md.label[i]];
					F.wsum += F.wi[i];
					++F.valid;
				}
			}
			gas.spend(kilo(size_t(end - F.cur) * 512));
			F.cur = end;
			if (F.cur == md.P) {
				F.state = Fit::S_ROUND_BEGIN;
			}
			return "";
		}
		case Fit::S_ROUND_BEGIN:
			for (uint32_t i = 0; i < md.P; ++i) {
				const double wn = F.wsum > 0.0 ? F.wi[i] / F.wsum : 0.0;
				F.w[i * 2] = float(wn);
				F.w[i * 2 + 1] = float(wn * F.W[W_POINT]);
			}
			{
				std::vector<uint8_t> in(md.P, 0);
				for (uint32_t i = 0; i < md.P; ++i) {
					in[i] = F.w[i * 2] != 0.0f || F.w[i * 2 + 1] != 0.0f;
				}
				for (uint32_t k = 0; k < F.K; ++k) {
					in[F.mid[k]] = 1;
				}
				std::vector<int32_t> at(md.P, -1);
				F.S.clear(), F.qS.clear(), F.nrmS.clear(), F.wS.clear(), F.midS.clear();
				for (uint32_t i = 0; i < md.P; ++i) {
					if (!in[i]) {
						continue;
					}
					at[i] = int32_t(F.S.size());
					F.S.push_back(i);
					F.qS.insert(F.qS.end(), &F.q[i * 3], &F.q[i * 3] + 3);
					F.nrmS.insert(F.nrmS.end(), &F.nrm[i * 3], &F.nrm[i * 3] + 3);
					F.wS.insert(F.wS.end(), &F.w[i * 2], &F.w[i * 2] + 2);
				}
				for (uint32_t k = 0; k < F.K; ++k) {
					F.midS.push_back(uint32_t(at[F.mid[k]]));
				}
				++F.sub_gen;
			}
			snapshot_problem(F);
			F.st = F.drv.start(F.n, F.x.data(), F.lb.data(), F.ub.data(), lbfgsb_params(F));
			F.xacc = F.x;
			F.facc = std::numeric_limits<double>::quiet_NaN();
			F.state = Fit::S_RUNNING;
			gas.spend(kilo(md.P));
			return fmt("RUNNING round %d start valid=%u support=%zu", F.round, F.valid, F.S.size());
		case Fit::S_RUNNING:
			switch (F.st) {
				case Lbfgsb::BUSY:
					F.st = F.drv.next();
					gas.spend(1);
					return "";
				case Lbfgsb::NEED_EVAL:
				case Lbfgsb::TRY: {
					std::vector<float> xs;
					if (!F.drv.readX(xs)) {
						return fit_fail(F, "readX: " + F.vec.error());
					}
					pipe_start(F.pipe, md, xs, true, F.n, F.K, uint32_t(F.S.size()));
					F.state = Fit::S_EVAL;
					gas.spend(1);
					return "";
				}
				case Lbfgsb::ACCEPT: {
					++F.accepts;
					F.drv.readX(F.xacc);
					F.facc = F.drv.fx();
					const std::string s = fmt("RUNNING round %d iter %d f %.9g pg %.3g", F.round, F.drv.iterations(),
							F.drv.fx(), F.drv.pgNorm());
					F.st = F.drv.next();
					gas.spend(1);
					return s;
				}
				case Lbfgsb::CONVERGED: {
					F.drv.readX(F.x);
					gas.spend(1);
					return round_end(F, F.drv.fx(), F.drv.reason());
				}
				case Lbfgsb::FAIL:
				default: {
					// A line search that cannot decrease f further has reached the
					// float32 loss's resolution (the identity fit starts there): the
					// round keeps its last accepted iterate and says it stalled.
					const std::string e = F.drv.error();
					if (e.find("line search") == std::string::npos) {
						return fit_fail(F, "lbfgsb: " + e);
					}
					F.x = F.xacc;
					++F.stalls;
					gas.spend(1);
					return round_end(F, std::isnan(F.facc) ? F.drv.fx() : F.facc, "stalled (" + e + ")");
				}
			}
		case Fit::S_EVAL:
			if (pipe_item(F.pipe, F, gas)) {
				++F.evals;
				if (!std::isfinite(F.pipe.f)) {
					return fit_fail(F, "non-finite loss");
				}
				F.drv.setGradient(F.pipe.g.data());
				F.st = F.drv.next(F.pipe.f);
				F.state = Fit::S_RUNNING;
			}
			return "";
		case Fit::S_DONE:
			return fmt("DONE rounds %d f %.9g", F.round, F.f);
		default:
			return "FAIL " + F.err;
	}
}

std::string fit_tick(Fit &F, Gas &gas) {
	while (gas.more()) {
		const std::string s = fit_item(F, gas);
		if (!s.empty()) {
			return s;
		}
	}
	return fmt("RUNNING round %d state %d evals %d accepts %d", F.round, int(F.state), F.evals, F.accepts);
}

// The rotation of x's rot6 (its first two rows, Gram-Schmidt re-orthonormalised,
// the third their cross product; AGENTS.md rule 11), row-major, in double.
void rot6_matrix(const float *x, double R[9]) {
	const double a1[3] = { x[0], x[1], x[2] }, a2[3] = { x[3], x[4], x[5] };
	const double n1 = std::max(std::sqrt(a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2]), 1e-20);
	const double r1[3] = { a1[0] / n1, a1[1] / n1, a1[2] / n1 };
	const double d = r1[0] * a2[0] + r1[1] * a2[1] + r1[2] * a2[2];
	const double u[3] = { a2[0] - d * r1[0], a2[1] - d * r1[1], a2[2] - d * r1[2] };
	const double nu = std::max(std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), 1e-20);
	const double r2[3] = { u[0] / nu, u[1] / nu, u[2] / nu };
	const double r3[3] = { r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0] };
	for (int k = 0; k < 3; ++k) {
		R[k] = r1[k], R[3 + k] = r2[k], R[6 + k] = r3[k];
	}
}

// A mask's weight at model vertex i (model frame, mm): a smooth step 16 mm
// wide across the midline (ANNY's +x is the character's left) or 6 mm wide
// across the lip line. left + right = 1 and upper + lower = 1 everywhere.
double mask_weight(const Model &md, uint32_t i, const std::string &m) {
	const double x = md.base[i * 3], y = md.base[i * 3 + 1];
	if (m == "left") {
		return smooth01(x / 8.0);
	}
	if (m == "right") {
		return 1.0 - smooth01(x / 8.0);
	}
	if (m == "upper") {
		return smooth01((y - md.y_lip) / 3.0);
	}
	return 1.0 - smooth01((y - md.y_lip) / 3.0); // lower
}

struct Fields {
	std::vector<float> dm; // P*3 mm, ANNY's model space
	std::vector<double> art; // N*3 mm, the avatar's own shapes
	int actions = 0, units = 0, artists = 0; // actions counts every model-space term (units too)
};

// dm += scale * prod(masks) * src, per model vertex (src P*3, model space).
void add_model(const Model &md, const float *src, const Term &t, std::vector<float> &dm) {
	for (uint32_t i = 0; i < md.P; ++i) {
		double m = t.scale;
		for (const std::string &mk : t.masks) {
			m *= mask_weight(md, i, mk);
		}
		for (int k = 0; k < 3; ++k) {
			dm[i * 3 + k] = float(double(dm[i * 3 + k]) + m * double(src[i * 3 + k]));
		}
	}
}

// Add a sum of terms into f. False with the reason for a term dress-on does
// not compute (procedural, corrective, bone, baked, absent) or cannot find.
bool term_fields(Fit &F, const std::vector<Entry> &map, const std::map<std::string, size_t> &by,
		const std::vector<Term> &terms, Fields &f, std::string &why, int depth) {
	const Model &md = *F.md;
	if (depth > 16) {
		why = "the sum nests deeper than 16 (a cycle?)";
		return false;
	}
	for (const Term &t : terms) {
		switch (t.k) {
			case Term::ACTION: {
				const int a = facial_index(t.ref);
				if (a < 0) {
					why = "`" + t.ref + "` is neither one of ANNY's 52 actions nor a name of the map";
					return false;
				}
				add_model(md, &md.fa_d[size_t(a) * md.P * 3], t, f.dm);
				++f.actions;
				break;
			}
			case Term::UNIT: {
				// The race fold, as the macro rows fold race: 1/3 of each race's file.
				std::vector<float> row(size_t(md.P) * 3, 0.0f);
				for (const char *race : kRaces) {
					const TFile *tf = need(unit_key(race, t.ref));
					if (!tf) {
						why = g_err;
						return false;
					}
					add_into(row.data(), *tf, md.c2r, kRaceFold);
				}
				add_model(md, row.data(), t, f.dm);
				++f.actions, ++f.units;
				break;
			}
			case Term::ENTRY: {
				const Entry &e = map[by.at(t.ref)];
				Fields sub;
				sub.dm.assign(f.dm.size(), 0.0f);
				sub.art.assign(f.art.size(), 0.0);
				if (!term_fields(F, map, by, e.terms, sub, why, depth + 1)) {
					why = t.ref + ": " + why;
					return false;
				}
				if (sub.artists && !t.masks.empty()) {
					why = t.ref + " * mask: masks apply to ANNY's model space, and " + t.ref + " has an artist term";
					return false;
				}
				add_model(md, sub.dm.data(), t, f.dm);
				for (size_t k = 0; k < f.art.size(); ++k) {
					f.art[k] += t.scale * sub.art[k];
				}
				f.actions += sub.actions, f.units += sub.units, f.artists += sub.artists;
				break;
			}
			case Term::ARTIST: {
				const Mesh &ms = *F.ms;
				size_t s = ms.shape_names.size();
				for (size_t k = 0; k < ms.shape_names.size(); ++k) {
					if (ms.shape_names[k] == t.ref) {
						s = k;
					}
				}
				if (s == ms.shape_names.size()) {
					why = fmt("artist:%s is not among the %zu shapes the host added to this mesh (hf_mesh_add_shape)",
							t.ref.c_str(), ms.shape_names.size());
					return false;
				}
				for (size_t k = 0; k < f.art.size(); ++k) {
					f.art[k] += t.scale * double(ms.shapes[s][k]) * kMToMm;
				}
				++f.artists;
				break;
			}
			case Term::PROCEDURAL:
				why = "procedural:" + t.ref + " is the next stage (RFD 2275's tongue frame), not computed here";
				return false;
			case Term::CORRECTIVE:
				why = "corrective:" + t.ref + " is the next stage (RFD 2275/2279), not computed here";
				return false;
			case Term::BONE:
				why = "bone:" + t.ref + " drives a bone (RFD 2279), not a shape";
				return false;
			case Term::BAKED:
				why = "baked:" + t.ref + " is baked from bone poses (RFD 2279), not computed here";
				return false;
			case Term::MATERIAL:
				why = "material:" + t.ref + " drives a material parameter, not a shape; the host sets it";
				return false;
			case Term::ABSENT:
			default:
				why = "absent: " + t.ref;
				return false;
		}
	}
	return true;
}


// ---- reports ------------------------------------------------------------------

struct RegionStat {
	uint32_t n = 0, valid = 0;
	double rms = 0, mean = 0, p95 = 0, max = 0;
};

// The model posed at the fit's current x (a pipe of its own, no gas: reports
// are one pass), and its normals.
struct Posed {
	Pipe p;
	std::vector<hfg::V3> mn;
};
void pose_now(const Fit &F, Posed &out) {
	pipe_start(out.p, *F.md, F.x, false, F.n, F.K);
	pipe_run(out.p, F);
	out.mn = model_normals(*F.md, out.p.pos);
}

RegionStat region_stat(const Fit &F, const Posed &ps, int region) {
	const Model &md = *F.md;
	std::vector<double> d;
	RegionStat s;
	for (uint32_t i = 0; i < md.P; ++i) {
		if (region == R_ALL ? md.label[i] == R_NECK : md.label[i] != region) {
			continue;
		}
		++s.n;
		const Corr c = correspond(F, ps.p.pos, i, ps.mn, 1e30);
		if (!c.ok) {
			continue;
		}
		d.push_back(std::sqrt(c.c.d2));
	}
	s.valid = uint32_t(d.size());
	if (d.empty()) {
		return s;
	}
	double ss = 0, sm = 0;
	for (double x : d) {
		ss += x * x, sm += x;
	}
	s.rms = std::sqrt(ss / d.size());
	s.mean = sm / d.size();
	std::sort(d.begin(), d.end());
	s.p95 = d[std::min(d.size() - 1, size_t(0.95 * d.size()))];
	s.max = d.back();
	return s;
}

// ---- the transfer's hits: made once per fit, in items ----------------------------

// One item; true when the hits are ready. Only on a finished fit.
bool hits_item(Fit &F, Gas &gas) {
	const Model &md = *F.md;
	switch (F.hst) {
		case Fit::H_NONE:
			pipe_start(F.rp, md, F.x, false, F.n, F.K);
			F.hst = Fit::H_POSE;
			gas.spend(1);
			return false;
		case Fit::H_POSE:
			if (pipe_item(F.rp, F, gas)) {
				F.hst = Fit::H_NORMALS;
			}
			return false;
		case Fit::H_NORMALS:
			F.fpos = F.rp.pos;
			F.fmn = model_normals(md, F.fpos);
			F.pmesh.v.resize(md.P);
			for (uint32_t i = 0; i < md.P; ++i) {
				F.pmesh.v[i] = hfg::V3(F.fpos[i * 3], F.fpos[i * 3 + 1], F.fpos[i * 3 + 2]);
			}
			F.pmesh.t = md.tri;
			gas.spend(kilo(size_t(md.tri.size()) * 8));
			F.hst = Fit::H_BVH;
			return false;
		case Fit::H_BVH: {
			F.pbvh.build(F.pmesh);
			const uint32_t N = uint32_t(F.ms->m.v.size());
			F.hit.assign(N, 0xFFFFFFFFu);
			F.bary.assign(size_t(N) * 3, 0.0f);
			F.matched = 0;
			F.cur = 0;
			gas.spend(kilo(F.pmesh.nt() * 24));
			F.hst = Fit::H_RAYS;
			return false;
		}
		case Fit::H_RAYS: {
			// 64 avatar vertices: each casts along its normal, both ways, to the
			// fitted surface; a surface facing away is rejected, a miss stays
			// unmatched.
			const uint32_t N = uint32_t(F.ms->m.v.size());
			const uint32_t end = std::min(N, F.cur + 64);
			for (uint32_t j = F.cur; j < end; ++j) {
				if (hfg::norm(F.tnorm[j]) == 0.0) {
					continue;
				}
				// RFD 2275: "a normal test rejects surfaces facing away" -- dot < 0.
				// A stricter cone (0.5, 60 degrees) rejected every triangle around a
				// vertex on a crease, and the ray took the other lip's deltas
				// (gates/11-headfit, the identity gate).
				const hfg::Bvh::Hit h = F.pbvh.ray(F.ms->m.v[j], F.tnorm[j], F.W[W_MAXDIST], 0.0);
				if (h.tri < 0) {
					continue;
				}
				F.hit[j] = uint32_t(h.tri);
				for (int k = 0; k < 3; ++k) {
					F.bary[j * 3 + k] = float(h.bary[k]);
				}
				++F.matched;
			}
			gas.spend(kilo(size_t(end - F.cur) * 512));
			F.cur = end;
			if (F.cur == N) {
				F.hst = Fit::H_READY;
				return true;
			}
			return false;
		}
		case Fit::H_READY:
		default:
			return true;
	}
}

// ---- the transfer job -----------------------------------------------------------

struct Xfer {
	std::shared_ptr<Fit> F;
	std::string shape;
	bool field = false; // a given model-space field (the identity gate), not a name
	enum State { X_FIELDS, X_HITS, X_RESAMPLE, X_ERROR, X_FINISH, X_DONE, X_FAILED } st = X_FIELDS;
	Fields fd;
	std::vector<float> out; // N*3 mm
	uint32_t cur = 0;
	uint32_t support = 0, moved = 0, reached = 0, art_moved = 0;
	double ess = 0, mss = 0;
	double ess_r = 0, mss_r = 0; // over the reached vertices only
	double R[9];
	std::string form = "action", info, err;
	float error_mm = 0.0f;
	std::vector<float> deltas; // N*3 metres
};

std::string xfer_fail(Xfer &X, const std::string &why) {
	X.st = Xfer::X_FAILED;
	X.err = why;
	return "FAIL " + why;
}

std::string xfer_item(Xfer &X, Gas &gas) {
	Fit &F = *X.F;
	const Model &md = *F.md;
	const uint32_t N = uint32_t(F.ms->m.v.size());
	switch (X.st) {
		case Xfer::X_FIELDS: {
			if (!X.field) {
				// One of ANNY's 52 actions, or a unified name through the map.
				std::vector<Entry> map;
				std::map<std::string, size_t> by;
				std::vector<Term> terms;
				if (facial_index(X.shape) >= 0) {
					Term t;
					t.ref = X.shape;
					terms.push_back(t);
				} else {
					std::string errs, unknown;
					uint32_t nunknown = 0;
					if (!parse_map(map, errs)) {
						return xfer_fail(X, g_err);
					}
					resolve_map(map, unknown, nunknown);
					for (size_t k = 0; k < map.size(); ++k) {
						by[map[k].name] = k;
					}
					auto it = by.find(X.shape);
					if (it == by.end()) {
						return xfer_fail(X, fmt("hf_transfer: %s is neither an ANNY action nor a name in unified_expressions.map",
													X.shape.c_str()));
					}
					terms = map[it->second].terms;
					X.form = map[it->second].form;
				}
				X.fd.dm.assign(size_t(md.P) * 3, 0.0f);
				X.fd.art.assign(size_t(N) * 3, 0.0);
				std::string why;
				if (!term_fields(F, map, by, terms, X.fd, why, 0)) {
					return xfer_fail(X, fmt("hf_transfer: %s refused: %s", X.shape.c_str(), why.c_str()));
				}
			}
			for (uint32_t i = 0; i < md.P; ++i) {
				if (X.fd.dm[i * 3] != 0.0f || X.fd.dm[i * 3 + 1] != 0.0f || X.fd.dm[i * 3 + 2] != 0.0f) {
					++X.support;
				}
			}
			X.out.assign(size_t(N) * 3, 0.0f);
			gas.spend(kilo(size_t(md.P) * 3 * 8));
			X.st = X.fd.actions > 0 ? Xfer::X_HITS : Xfer::X_FINISH;
			return "";
		}
		case Xfer::X_HITS:
			if (hits_item(F, gas)) {
				X.st = Xfer::X_RESAMPLE;
				X.cur = 0;
			}
			return "";
		case Xfer::X_RESAMPLE: {
			// The barycentric blend of the model's deltas at each hit, through
			// the fit's s R (hf_resample), 64 avatar vertices an item.
			float prm[10];
			prm_of(F.x.data(), prm);
			hfk::resample(N, prm, md.tri.data(), F.hit.data(), F.bary.data(), X.fd.dm.data(), X.out.data(), X.cur,
					X.cur + 1);
			gas.spend(1);
			if (++X.cur == (N + 63) / 64) {
				for (uint32_t j = 0; j < N; ++j) {
					if (X.out[j * 3] != 0.0f || X.out[j * 3 + 1] != 0.0f || X.out[j * 3 + 2] != 0.0f) {
						++X.moved;
					}
				}
				if (X.support == 0 || X.moved == 0) {
					return xfer_fail(X, fmt("hf_transfer: %s refused: empty support (%u model vertices move, %u avatar "
											"vertices matched to them); at weight 0 it would look correct",
												X.shape.c_str(), X.support, X.moved));
				}
				rot6_matrix(F.x.data(), X.R);
				X.st = Xfer::X_ERROR;
				X.cur = 0;
			}
			return "";
		}
		case Xfer::X_ERROR: {
			// The round trip, 64 model vertices an item: each supported model
			// vertex's closest avatar point blends the transferred deltas back,
			// against the model's own delta carried through s R. Artist terms
			// are the avatar's own and add no error; they are left out of it.
			const float *x = F.x.data();
			const uint32_t end = std::min(md.P, X.cur + 64);
			for (uint32_t i = X.cur; i < end; ++i) {
				const float *d = &X.fd.dm[i * 3];
				if (d[0] == 0.0f && d[1] == 0.0f && d[2] == 0.0f) {
					continue;
				}
				double sd[3];
				for (int r = 0; r < 3; ++r) {
					sd[r] = x[9] * (X.R[r * 3] * d[0] + X.R[r * 3 + 1] * d[1] + X.R[r * 3 + 2] * d[2]);
				}
				X.mss += sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2];
				const Corr c = correspond(F, F.fpos, i, F.fmn, F.W[W_MAXDIST]);
				double back[3] = { 0, 0, 0 };
				if (c.ok) {
					++X.reached;
					for (int k = 0; k < 3; ++k) {
						const uint32_t tv = F.ms->m.t[size_t(c.c.tri) * 3 + k];
						for (int r = 0; r < 3; ++r) {
							back[r] += c.c.bary[k] * X.out[tv * 3 + r];
						}
					}
				}
				for (int r = 0; r < 3; ++r) {
					X.ess += (back[r] - sd[r]) * (back[r] - sd[r]);
				}
				if (c.ok) {
					for (int r = 0; r < 3; ++r) {
						X.ess_r += (back[r] - sd[r]) * (back[r] - sd[r]);
						X.mss_r += sd[r] * sd[r];
					}
				}
			}
			gas.spend(kilo(size_t(end - X.cur) * 512));
			X.cur = end;
			if (X.cur == md.P) {
				X.st = Xfer::X_FINISH;
			}
			return "";
		}
		case Xfer::X_FINISH: {
			for (uint32_t j = 0; j < N; ++j) {
				const double *a = &X.fd.art[j * 3];
				if (a[0] != 0.0 || a[1] != 0.0 || a[2] != 0.0) {
					++X.art_moved;
				}
			}
			if (X.fd.artists > 0 && X.art_moved == 0) {
				return xfer_fail(X, fmt("hf_transfer: %s refused: its artist shapes move no vertex; at weight 0 it would "
										"look correct",
											X.shape.c_str()));
			}
			X.error_mm = X.support ? float(std::sqrt(X.ess / X.support)) : 0.0f;
			const double mag = X.support ? std::sqrt(X.mss / X.support) : 0.0;
			const bool hits = F.hst == Fit::H_READY;
			X.info = fmt("shape %s form %s actions %d units %d artists %d support %u reached %u unreached %u "
						 "avatar_moved %u artist_moved %u avatar_matched %u avatar_unmatched %u magnitude_mm %.4f "
						 "error_mm %.4f rel %.4f reached_error_mm %.4f reached_rel %.4f",
					X.shape.c_str(), X.form.c_str(), X.fd.actions - X.fd.units, X.fd.units, X.fd.artists, X.support,
					X.reached, X.support - X.reached, X.moved, X.art_moved, hits ? F.matched : 0u,
					hits ? N - F.matched : 0u, mag, double(X.error_mm), mag > 0 ? double(X.error_mm) / mag : 0.0, X.reached ? std::sqrt(X.ess_r / X.reached) : 0.0,
					X.mss_r > 0 ? std::sqrt(X.ess_r / X.mss_r) : 0.0);
			X.deltas.resize(X.out.size());
			for (size_t k = 0; k < X.out.size(); ++k) {
				X.deltas[k] = float((double(X.out[k]) + X.fd.art[k]) / kMToMm);
			}
			X.out.clear(), X.out.shrink_to_fit();
			X.fd = Fields();
			gas.spend(kilo(size_t(N) * 3));
			X.st = Xfer::X_DONE;
			return "DONE " + X.info;
		}
		case Xfer::X_DONE:
			return "DONE " + X.info;
		default:
			return "FAIL " + X.err;
	}
}

std::string xfer_tick(Xfer &X, Gas &gas) {
	while (gas.more()) {
		const std::string s = xfer_item(X, gas);
		if (!s.empty()) {
			return s;
		}
	}
	return fmt("RUNNING transfer %s state %d item %u", X.shape.c_str(), int(X.st), X.cur);
}


} // namespace

// ---- the public surface -------------------------------------------------------

void data_put(const std::string &key, const std::string &text) {
	g_err.clear();
	if (key == kBaseKey || is_target_key(key)) {
		Parse p;
		p.key = key;
		p.text = text;
		p.base = key == kBaseKey;
		parse_queue().push_back(std::move(p));
		return;
	}
	raw()[key] = text;
}

bool data_has(const std::string &key) {
	if (raw().count(key) || tfiles().count(key) || (key == kBaseKey && g_base.ok)) {
		return true;
	}
	for (const Parse &p : parse_queue()) {
		if (p.key == key) {
			return true;
		}
	}
	return false;
}

size_t data_count() {
	return raw().size() + tfiles().size() + (g_base.ok ? 1 : 0) + parse_queue().size();
}

size_t data_queued() {
	return parse_queue().size();
}

void data_clear() {
	raw().clear();
	tfiles().clear();
	parse_queue().clear();
	g_base = Base();
	g_load_err.clear();
	g_parsed = 0;
}

const std::string &last_error() {
	return g_err;
}

std::string data_report() {
	std::string s = fmt("base verts %u faces %u quads %u tris %u body_tris %zu y_neck_dm %.6f neck_cube_verts %u "
						"candidates %zu queued %zu parsed %llu load_error %s\n",
			g_base.nv, g_base.nfaces, g_base.nquads, g_base.ntris, g_base.body_tris.size() / 3, g_base.y_neck,
			g_base.neck_cube_verts, g_base.cand_ids.size(), parse_queue().size(), (unsigned long long)g_parsed,
			g_load_err.empty() ? "none" : g_load_err.c_str());
	for (const auto &kv : tfiles()) {
		s += fmt("target %s lines %llu idx_sum %lld idx_max %lld sum %.17g abs_sum %.17g kept %zu dropped_moving %u\n",
				kv.first.c_str(), (unsigned long long)kv.second.lines, (long long)kv.second.idx_sum,
				(long long)kv.second.idx_max, kv.second.sum, kv.second.abs_sum, kv.second.ci.size(),
				kv.second.dropped_moving);
	}
	return s;
}

int mesh_create(const float *xyz, int nv, const int32_t *tri, int nt) {
	if (nv < 3 || nt < 1) {
		fail("hf_mesh_create: %d vertices, %d triangles", nv, nt);
		return -1;
	}
	auto m = std::make_shared<Mesh>();
	m->m.v.resize(size_t(nv));
	for (int i = 0; i < nv; ++i) {
		for (int k = 0; k < 3; ++k) {
			if (!std::isfinite(xyz[i * 3 + k])) {
				fail("hf_mesh_create: vertex %d is not finite", i);
				return -1;
			}
		}
		m->m.v[size_t(i)] =
				hfg::V3(double(xyz[i * 3]) * kMToMm, double(xyz[i * 3 + 1]) * kMToMm, double(xyz[i * 3 + 2]) * kMToMm);
	}
	for (int k = 0; k < 3; ++k) {
		double s = 0.0;
		for (int i = 0; i < nv; ++i) {
			s += double(xyz[i * 3 + k]) * kMToMm;
		}
		m->origin[k] = s / double(nv);
	}
	for (int i = 0; i < nv; ++i) {
		m->m.v[size_t(i)] = m->m.v[size_t(i)] - hfg::V3(m->origin[0], m->origin[1], m->origin[2]);
	}
	m->m.t.resize(size_t(nt) * 3);
	for (int k = 0; k < nt * 3; ++k) {
		if (tri[k] < 0 || tri[k] >= nv) {
			fail("hf_mesh_create: triangle index %d out of range", tri[k]);
			return -1;
		}
		m->m.t[size_t(k)] = uint32_t(tri[k]);
	}
	g_err.clear();
	return put(K_MESH, m);
}

int mesh_add_shape(int mesh, const std::string &name, const float *deltas, int n3) {
	Mesh *m = get<Mesh>(mesh, K_MESH, "mesh");
	if (!m) {
		return -1;
	}
	if (size_t(n3) != m->m.v.size() * 3) {
		fail("hf_mesh_add_shape: %d floats for %zu vertices", n3, m->m.v.size());
		return -1;
	}
	for (const std::string &s : m->shape_names) {
		if (s == name) {
			fail("hf_mesh_add_shape: %s already added", name.c_str());
			return -1;
		}
	}
	for (int k = 0; k < n3; ++k) {
		if (!std::isfinite(deltas[k])) {
			fail("hf_mesh_add_shape: %s has a non-finite delta", name.c_str());
			return -1;
		}
	}
	m->shape_names.push_back(name);
	m->shapes.emplace_back(deltas, deltas + n3);
	g_err.clear();
	return int(m->shape_names.size()) - 1;
}

int model_load(int region) {
	if (region != 0 && region != 1) {
		fail("hf_model_load: region 0 (the head) or 1 (the face)");
		return -1;
	}
	auto md = std::make_shared<Model>();
	md->region = region;
	g_err.clear();
	return put(K_MODEL, md);
}

int marks_create(const int32_t *model_ids, const int32_t *avatar_ids, int n) {
	if (n < 3) {
		fail("hf_marks_create: %d marks (at least 3 place the similarity)", n);
		return -1;
	}
	auto mk = std::make_shared<Marks>();
	for (int k = 0; k < n; ++k) {
		for (int j = 0; j < k; ++j) {
			if (model_ids[j] == model_ids[k]) {
				fail("hf_marks_create: model vertex %d marked twice", model_ids[k]);
				return -1;
			}
		}
		mk->model_ids.push_back(model_ids[k]);
		mk->avatar_ids.push_back(avatar_ids[k]);
	}
	g_err.clear();
	return put(K_MARKS, mk);
}

// A model the fit or a report can use: built, and on the base.obj now loaded.
Model *ready_model(int model, const char *who) {
	Model *md = get<Model>(model, K_MODEL, "model");
	if (!md) {
		return nullptr;
	}
	if (!md->ready()) {
		fail("%s: model %d is not built yet (tick it until DONE)", who, model);
		return nullptr;
	}
	if (md->base_gen != g_base_gen) {
		fail("%s: model %d was built on another base.obj", who, model);
		return nullptr;
	}
	return md;
}

int solve(int model, int mesh, int marks, const float *w, int nw) {
	if (!ready_model(model, "hf_solve") || !get<Mesh>(mesh, K_MESH, "mesh") || !get<Marks>(marks, K_MARKS, "marks")) {
		return -1;
	}
	auto F = std::make_shared<Fit>();
	F->md = std::static_pointer_cast<Model>(handles()[model].p);
	F->ms = std::static_pointer_cast<Mesh>(handles()[mesh].p);
	F->mk = std::static_pointer_cast<Marks>(handles()[marks].p);
	for (int k = 0; k < W_COUNT; ++k) {
		F->W[k] = k < nw ? w[k] : kWeightDefaults[k];
		if (!std::isfinite(F->W[k]) || F->W[k] < 0.0f) {
			fail("hf_solve: weight %d is %g (finite, >= 0)", k, double(F->W[k]));
			return -1;
		}
	}
	const Model &md = *F->md;
	F->rounds = std::max(1, int(F->W[W_ROUNDS]));
	// Landmarks: base.obj ids to region ids.
	std::vector<int32_t> g2r(g_base.nv, -1);
	for (uint32_t r = 0; r < md.P; ++r) {
		g2r[md.gid[r]] = int32_t(r);
	}
	F->K = uint32_t(F->mk->model_ids.size());
	for (uint32_t k = 0; k < F->K; ++k) {
		const int32_t gm = F->mk->model_ids[k], ga = F->mk->avatar_ids[k];
		if (gm < 0 || uint32_t(gm) >= g_base.nv || g2r[size_t(gm)] < 0) {
			fail("hf_solve: mark %u: model vertex %d is not in the model's region", k, gm);
			return -1;
		}
		if (ga < 0 || size_t(ga) >= F->ms->m.v.size()) {
			fail("hf_solve: mark %u: avatar vertex %d out of range", k, ga);
			return -1;
		}
		F->mid.push_back(uint32_t(g2r[size_t(gm)]));
		const hfg::V3 &a = F->ms->m.v[size_t(ga)];
		F->tgtK.push_back(float(a.x)), F->tgtK.push_back(float(a.y)), F->tgtK.push_back(float(a.z));
		F->lw.push_back(F->W[W_LANDMARK] / float(F->K));
	}
	// Unknowns: rot6 | t | s | six axes | the dials' positive and negative rows.
	F->n = 16 + 2 * md.L;
	F->x.assign(F->n, 0.0f);
	F->lb.assign(F->n, -INFINITY);
	F->ub.assign(F->n, INFINITY);
	F->px0.assign(F->n, 0.0f);
	F->plam.assign(F->n, 0.0f);
	for (int a = 0; a < 6; ++a) {
		// In x's units (kPScale): the same prior lam (phen - 0.5)^2 and bounds.
		F->x[10 + a] = 0.5f / kPScale; // ANNY's default phenotype
		F->px0[10 + a] = 0.5f / kPScale;
		F->plam[10 + a] = F->W[W_PRIOR_PHEN] * kPScale * kPScale;
		F->lb[10 + a] = float(kAnchors[a][0] / double(kPScale));
		F->ub[10 + a] = float(kAnchors[a][kAxisCount[a] - 1] / double(kPScale));
	}
	for (uint32_t k = 0; k < 2 * md.L; ++k) {
		F->lb[16 + k] = 0.0f;
		F->ub[16 + k] = 1.0f;
		F->plam[16 + k] = F->W[W_PRIOR_DIAL];
	}
	F->lb[9] = 0.01f;
	F->ub[9] = 100.0f;
	F->q.resize(size_t(md.P) * 3);
	F->nrm.resize(size_t(md.P) * 3);
	F->w.resize(size_t(md.P) * 2);
	F->state = Fit::S_BVH;
	g_err.clear();
	return put(K_FIT, F);
}

std::string tick(int handle, int64_t budget) {
	Gas gas{ budget };
	if (handle == 0) {
		const std::string s = loader_tick(gas);
		if (!s.empty()) {
			return s;
		}
		return !g_load_err.empty() ? "FAIL " + g_load_err : fmt("DONE loader parsed %llu", (unsigned long long)g_parsed);
	}
	auto it = handles().find(handle);
	if (it == handles().end()) {
		fail("hf_tick: %d is not a live handle", handle);
		return "FAIL " + g_err;
	}
	switch (it->second.kind) {
		case K_MODEL:
			return model_tick(*static_cast<Model *>(it->second.p.get()), gas);
		case K_FIT: {
			Fit &F = *static_cast<Fit *>(it->second.p.get());
			if (F.md->base_gen != g_base_gen) {
				return fit_fail(F, "the model's base.obj was replaced");
			}
			return fit_tick(F, gas);
		}
		case K_XFER:
			return xfer_tick(*static_cast<Xfer *>(it->second.p.get()), gas);
		default:
			return fmt("DONE %d needs no ticks", handle);
	}
}

bool done(int handle) {
	auto it = handles().find(handle);
	if (it == handles().end()) {
		return false;
	}
	switch (it->second.kind) {
		case K_MODEL:
			return static_cast<Model *>(it->second.p.get())->ready();
		case K_FIT:
			return static_cast<Fit *>(it->second.p.get())->state == Fit::S_DONE;
		case K_XFER:
			return static_cast<Xfer *>(it->second.p.get())->st == Xfer::X_DONE;
		default:
			return true;
	}
}

float fit_residual_mm(int fit, int region) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return -1.0f;
	}
	if (region < 0 || region >= R_COUNT) {
		fail("hf_fit_residual_mm: region %d", region);
		return -1.0f;
	}
	Posed ps;
	pose_now(*F, ps);
	return float(region_stat(*F, ps, region).rms);
}

std::string fit_report(int fit) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return "FAIL " + g_err;
	}
	const Model &md = *F->md;
	std::string s = fmt("fit state %s rounds %d evals %d accepts %d stalls %d n %u P %u NB %u active_rows %zu K %u f %.9g\n",
			F->state == Fit::S_DONE ? "DONE" : (F->state == Fit::S_FAILED ? "FAILED" : "RUNNING"), F->round, F->evals,
			F->accepts, F->stalls, F->n, md.P, md.NB, F->pipe.act.size(), F->K, F->f);
	s += fmt("x rot6 %.7g %.7g %.7g %.7g %.7g %.7g t %.7g %.7g %.7g s %.7g\n", F->x[0], F->x[1], F->x[2], F->x[3],
			F->x[4], F->x[5], F->x[6] * kTScale, F->x[7] * kTScale, F->x[8] * kTScale, F->x[9]);
	s += fmt("t_avatar_mm %.4f %.4f %.4f (t plus the mesh origin %.4f %.4f %.4f)\n", F->x[6] * kTScale + F->ms->origin[0],
			F->x[7] * kTScale + F->ms->origin[1], F->x[8] * kTScale + F->ms->origin[2], F->ms->origin[0], F->ms->origin[1], F->ms->origin[2]);
	s += "phenotype";
	for (int a = 0; a < 6; ++a) {
		s += fmt(" %s %.6f", kAxes[a], F->x[10 + a] * kPScale);
	}
	s += "\n";
	for (size_t r = 0; r < F->round_f.size(); ++r) {
		s += fmt("round %zu f %.9g iters %d\n", r, F->round_f[r], F->round_iters[r]);
	}
	std::vector<std::pair<float, std::string>> dl;
	for (uint32_t k = 0; k < md.L; ++k) {
		const float v = F->x[16 + 2 * k] - F->x[16 + 2 * k + 1];
		if (std::fabs(v) > 0.02f) {
			dl.push_back({ -std::fabs(v), fmt("%s %+.3f", md.dial_names[k].c_str(), v) });
		}
	}
	std::sort(dl.begin(), dl.end());
	s += fmt("dials moved %zu:", dl.size());
	for (const auto &d : dl) {
		s += " " + d.second + ";";
	}
	s += "\n";
	Posed ps;
	pose_now(*F, ps);
	for (int r = 0; r < R_COUNT; ++r) {
		const RegionStat st = region_stat(*F, ps, r);
		s += fmt("residual %s n %u valid %u rms_mm %.4f mean_mm %.4f p95_mm %.4f max_mm %.4f weight %.2f\n",
				region_name(r), st.n, st.valid, st.rms, st.mean, st.p95, st.max,
				r == R_ALL ? 1.0 : (r == R_NECK ? 0.0 : double(F->W[r - 1])));
	}
	double lss = 0;
	for (uint32_t k = 0; k < F->K; ++k) {
		double d2 = 0;
		for (int c = 0; c < 3; ++c) {
			const double d = ps.p.pos[F->mid[k] * 3 + c] - F->tgtK[k * 3 + c];
			d2 += d * d;
		}
		lss += d2;
		s += fmt("landmark %u model %d avatar %d dist_mm %.4f\n", k, F->mk->model_ids[k], F->mk->avatar_ids[k],
				std::sqrt(d2));
	}
	s += fmt("landmarks rms_mm %.4f\n", F->K ? std::sqrt(lss / F->K) : 0.0);
	return s;
}

std::string model_report(int model) {
	Model *md = ready_model(model, "hf_model_report");
	if (!md) {
		return "FAIL " + g_err;
	}
	uint32_t lab[R_COUNT] = { 0 };
	for (uint8_t l : md->label) {
		++lab[l];
	}
	std::string s = fmt("model region %d P %u tris %zu NB %u macro_rows %u dials %u facial %zu\n", md->region, md->P,
			md->tri.size() / 3, md->NB, md->macro_rows, md->L, md->fa_names.size());
	s += fmt("files macro %u lines %llu dial %u lines %llu facial %u lines %llu\n", md->files_macro,
			(unsigned long long)md->lines_macro, md->files_dial, (unsigned long long)md->lines_dial, md->files_facial,
			(unsigned long long)md->lines_facial);
	s += fmt("labels face %u mouth %u nose %u eyes %u cranium %u neck %u head %u\n", lab[R_FACE], lab[R_MOUTH],
			lab[R_NOSE], lab[R_EYES], lab[R_CRANIUM], lab[R_NECK], md->P_head);
	s += fmt("centre_mm %.4f %.4f %.4f y_lip_mm %.4f\n", md->centre[0], md->centre[1], md->centre[2], md->y_lip);
	for (size_t k = 0; k < md->fa_names.size(); ++k) {
		s += fmt("action %s file_entries %u region_support %u\n", md->fa_names[k].c_str(), md->fa_file_support[k],
				md->fa_region_support[k]);
	}
	return s;
}

bool macro_coefficients(int model, const double phen[6], std::vector<std::string> &names, std::vector<double> &c) {
	Model *md = ready_model(model, "macro_coefficients");
	if (!md) {
		return false;
	}
	// In double, as the GDScript reference computes (its floats are doubles).
	double sv[17];
	for (int a = 0; a < 6; ++a) {
		double w[5], dw[5];
		interp(phen[a], a, w, dw);
		for (int k = 0; k < kAxisCount[a]; ++k) {
			sv[kAxisFirst[a] + k] = w[k];
		}
	}
	names.clear(), c.clear();
	for (uint32_t t = 0; t < md->macro_rows; ++t) {
		double w = 1.0;
		for (int k = 0; k < 5; ++k) {
			const int s = md->row_slots[t * 5 + k];
			if (s >= 0) {
				w *= sv[s];
			}
		}
		names.push_back(md->row_names[t]);
		c.push_back(w);
	}
	return true;
}

int transfer(int fit, const std::string &shape) {
	auto it = handles().find(fit);
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return -1;
	}
	if (F->state != Fit::S_DONE) {
		fail("hf_transfer: fit %d is not done (tick it until DONE)", fit);
		return -1;
	}
	auto X = std::make_shared<Xfer>();
	X->F = std::static_pointer_cast<Fit>(it->second.p);
	X->shape = shape;
	g_err.clear();
	return put(K_XFER, X);
}

int transfer_field(int fit, const std::vector<float> &dm) {
	auto it = handles().find(fit);
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return -1;
	}
	if (F->state != Fit::S_DONE) {
		fail("transfer_field: fit %d is not done", fit);
		return -1;
	}
	if (dm.size() != size_t(F->md->P) * 3) {
		fail("transfer_field: %zu floats for %u vertices", dm.size(), F->md->P);
		return -1;
	}
	auto X = std::make_shared<Xfer>();
	X->F = std::static_pointer_cast<Fit>(it->second.p);
	X->shape = "(field)";
	X->form = "field";
	X->field = true;
	X->fd.dm = dm;
	X->fd.art.assign(F->ms->m.v.size() * 3, 0.0);
	X->fd.actions = 1;
	g_err.clear();
	return put(K_XFER, X);
}

bool transfer_result(int job, std::vector<float> &deltas, float &error_mm, std::string &info) {
	Xfer *X = get<Xfer>(job, K_XFER, "transfer");
	if (!X) {
		return false;
	}
	if (X->st == Xfer::X_FAILED) {
		return fail("%s", X->err.c_str());
	}
	if (X->st != Xfer::X_DONE) {
		return fail("transfer %d is not done (tick it until DONE)", job);
	}
	deltas = X->deltas;
	error_mm = X->error_mm;
	info = X->info;
	return true;
}

std::string map_audit(const std::string &required_names, int mesh) {
	std::vector<Entry> map;
	std::string errs, unknown;
	uint32_t nunknown = 0;
	if (!parse_map(map, errs)) {
		return "FAIL " + g_err;
	}
	resolve_map(map, unknown, nunknown);
	const Mesh *ms = nullptr;
	if (mesh > 0) {
		ms = get<Mesh>(mesh, K_MESH, "mesh");
		if (!ms) {
			return "FAIL " + g_err;
		}
	}
	std::map<std::string, const Entry *> by;
	std::string dup;
	uint32_t ndup = 0;
	for (const Entry &e : map) {
		if (by.count(e.name)) {
			dup += " " + e.name, ++ndup;
		}
		by[e.name] = &e;
	}
	std::vector<std::string> req;
	{
		std::istringstream is(required_names);
		std::string l;
		while (std::getline(is, l)) {
			l = trim(l);
			if (!l.empty() && l[0] != '#') {
				req.push_back(l);
			}
		}
		std::sort(req.begin(), req.end());
		req.erase(std::unique(req.begin(), req.end()), req.end());
	}
	std::string unmapped;
	uint32_t nun = 0;
	for (const std::string &r : req) {
		if (!by.count(r)) {
			unmapped += " " + r, ++nun;
		}
	}
	std::string extra;
	uint32_t nextra = 0;
	for (const Entry &e : map) {
		if (!std::binary_search(req.begin(), req.end(), e.name)) {
			extra += " " + e.name, ++nextra;
		}
	}
	// Artist shapes against the mesh the host built, when one is given.
	std::string art_missing;
	uint32_t nart = 0, nart_missing = 0;
	for (const Entry &e : map) {
		for (const Term &t : e.terms) {
			if (t.k != Term::ARTIST) {
				continue;
			}
			++nart;
			if (ms && std::find(ms->shape_names.begin(), ms->shape_names.end(), t.ref) == ms->shape_names.end()) {
				art_missing += " " + e.name + "(" + t.ref + ")", ++nart_missing;
			}
		}
	}
	std::string s = fmt("audit required %zu entries %zu unmapped %u extra %u duplicates %u parse_errors %u "
						"unknown_sources %u artist_terms %u artist_missing %s\n",
			req.size(), map.size(), nun, nextra, ndup, uint32_t(std::count(errs.begin(), errs.end(), '\n')), nunknown,
			nart, ms ? fmt("%u", nart_missing).c_str() : "unchecked");
	std::map<int, std::vector<const Entry *>> stages;
	for (const Entry &e : map) {
		stages[e.stage].push_back(&e);
	}
	for (const auto &st : stages) {
		std::map<std::string, uint32_t> forms;
		uint32_t absent = 0, next = 0;
		std::string al;
		for (const Entry *e : st.second) {
			++forms[e->form];
			if (e->form == "absent") {
				++absent;
				al += fmt(" %s(%s);", e->name.c_str(), e->terms[0].ref.c_str());
			}
			for (const Term &t : e->terms) {
				if (t.k == Term::PROCEDURAL || t.k == Term::CORRECTIVE || t.k == Term::BONE || t.k == Term::BAKED) {
					++next;
					break;
				}
			}
		}
		s += fmt("stage %d names %zu mapped %u absent %u next_stage %u forms", st.first, st.second.size(),
				uint32_t(st.second.size()) - absent, absent, next);
		for (const auto &f : forms) {
			s += fmt(" %s=%u", f.first.c_str(), f.second);
		}
		s += "\n";
		s += fmt("stage %d absent:%s\n", st.first, al.c_str());
	}
	s += "unmapped:" + unmapped + "\n";
	s += "extra:" + extra + "\n";
	s += "duplicates:" + dup + "\n";
	s += "artist_missing:" + art_missing + "\n";
	if (!unknown.empty()) {
		s += "unknown:\n" + unknown;
	}
	if (!errs.empty()) {
		s += "errors:\n" + errs;
	}
	return s;
}

bool destroy(int handle) {
	auto it = handles().find(handle);
	if (it == handles().end()) {
		return fail("hf_destroy: %d is not a live handle", handle);
	}
	handles().erase(it);
	return true;
}

int live_handles() {
	return int(handles().size());
}

bool fit_x(int fit, std::vector<float> &x) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return false;
	}
	x = F->x;
	return true;
}

bool round_problem(int fit, int round, RoundProblem &out) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return false;
	}
	if (round < 0 || size_t(round) >= F->round_prob.size()) {
		return fail("round %d not reached", round);
	}
	out = F->round_prob[size_t(round)];
	const Model &md = *F->md;
	out.base = md.base;
	out.blend = md.blend;
	out.row_slots.clear();
	for (uint32_t t = 0; t < md.macro_rows; ++t) {
		std::vector<int> sl;
		for (int k = 0; k < 5; ++k) {
			if (md.row_slots[t * 5 + k] >= 0) {
				sl.push_back(md.row_slots[t * 5 + k]);
			}
		}
		out.row_slots.push_back(sl);
	}
	out.anchors.clear();
	out.axis_slots.clear();
	for (int a = 0; a < 6; ++a) {
		out.anchors.emplace_back(kAnchors[a], kAnchors[a] + kAxisCount[a]);
		std::vector<int> s;
		for (int k = 0; k < kAxisCount[a]; ++k) {
			s.push_back(kAxisFirst[a] + k);
		}
		out.axis_slots.push_back(s);
	}
	return true;
}

bool round_result(int fit, int round, std::vector<float> &x, double &f, int &iters) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return false;
	}
	if (round < 0 || size_t(round) >= F->round_x.size()) {
		return fail("round %d not finished", round);
	}
	x = F->round_x[size_t(round)];
	f = F->round_f[size_t(round)];
	iters = F->round_iters[size_t(round)];
	return true;
}

} // namespace hf

namespace hf {

bool model_mesh(int model, const double phen[6], std::vector<float> &xyz_m, std::vector<int32_t> &tri,
		std::vector<int32_t> &gid) {
	Model *md = ready_model(model, "model_mesh");
	if (!md) {
		return false;
	}
	float ph[6];
	for (int a = 0; a < 6; ++a) {
		ph[a] = float(phen[a]);
	}
	std::vector<float> dials(size_t(2) * md->L, 0.0f), c, v(size_t(md->P) * 3);
	coefficients(*md, ph, dials.data(), c);
	hfk::blend(md->P, md->NB, md->base.data(), md->blend.data(), c.data(), v.data());
	xyz_m.resize(v.size());
	for (uint32_t i = 0; i < md->P; ++i) {
		for (int k = 0; k < 3; ++k) {
			xyz_m[i * 3 + k] = float((double(v[i * 3 + k]) + md->centre[k]) / kMToMm);
		}
	}
	tri.assign(md->tri.begin(), md->tri.end());
	gid.assign(md->gid.begin(), md->gid.end());
	return true;
}

bool model_action(int model, const std::string &action, std::vector<float> &d_m) {
	Model *md = ready_model(model, "model_action");
	if (!md) {
		return false;
	}
	const int a = facial_index(action);
	if (a < 0) {
		return fail("model_action: %s is not one of ANNY's 52 actions", action.c_str());
	}
	d_m.resize(size_t(md->P) * 3);
	for (size_t k = 0; k < d_m.size(); ++k) {
		d_m[k] = float(double(md->fa_d[size_t(a) * md->P * 3 + k]) / kMToMm);
	}
	return true;
}

} // namespace hf

namespace hf {

bool evaluate_at(int fit, const std::vector<float> &x, double &f, std::vector<float> &g) {
	Fit *F = get<Fit>(fit, K_FIT, "fit");
	if (!F) {
		return false;
	}
	if (F->S.empty() || x.size() != F->n) {
		return fail("evaluate_at: no round started, or x has %zu of %u unknowns", x.size(), F->n);
	}
	Pipe p;
	pipe_start(p, *F->md, x, true, F->n, F->K, uint32_t(F->S.size()));
	pipe_run(p, *F);
	f = p.f;
	g = p.g;
	return true;
}

bool model_blend(int model, std::vector<float> &base, std::vector<float> &blend) {
	Model *md = ready_model(model, "model_blend");
	if (!md) {
		return false;
	}
	base = md->base;
	blend = md->blend;
	return true;
}

} // namespace hf
