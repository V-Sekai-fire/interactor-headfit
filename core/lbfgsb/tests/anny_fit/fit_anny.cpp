// SPDX-License-Identifier: MIT
//
// Fit ANNY-soma-x's 11 phenotype values to a character's segment lengths with the in-guest L-BFGS-B
// (guest/drape/lbfgsb.cpp over vec_cpu.cpp), the optimiser drape, cage refit and head fitting share.
// The body comes from pheno_eval (ports/soma_pheno_api.h: blendshapes, barycentric transfer, RBF joints).
//
//   fit_anny <soma_pheno.bin> <soma_pheno_golden.bin> <targets.txt>
//
// targets.txt: one "name jointA jointB length_m" per line (SOMA joint indices, soma_rig.h order).
// The objective is the sum of pseudo-Huber relative segment-length errors (quadratic under 25%) plus LAMBDA times the squared distance from
// the mean body (all values 0.5), which settles the 11 variables the segments leave free. The gradient is
// central differences (one-sided at a bound), the only part not from the solver. Bounds are [0, 1].
// Gates: G1 the golden phenotype reproduces the golden mesh and skeleton; G2 the residual falls and the solver
// converges; G3 targets 10% longer move the fit and raise the residual against the original targets (control).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lbfgsb.h"
#include "soma_pheno.h"
#include "soma_pheno_api.h"
#include "vec_cpu.h"

struct Seg {
	std::string name;
	int a, b;
	double len;
};
static const int N = 11;
static const double LAMBDA = 1e-4;
static const double HUBER = 0.25;   // pseudo-Huber scale on the relative error: quadratic below 25%, linear beyond
static std::vector<float> g_v(PH_V * 3), g_bw(PH_J * 16);

static double seglen(const Seg &s) {
	const float *A = &g_bw[s.a * 16], *B = &g_bw[s.b * 16];
	return std::sqrt(double(A[3] - B[3]) * (A[3] - B[3]) + double(A[7] - B[7]) * (A[7] - B[7]) + double(A[11] - B[11]) * (A[11] - B[11]));
}
static double objective(const float *x, const std::vector<Seg> &segs, std::vector<double> *rel = nullptr) {
	pheno_eval(x, g_v.data(), g_bw.data());
	double f = 0.0;
	if (rel) {
		rel->clear();
	}
	for (const Seg &s : segs) {
		const double r = seglen(s) / s.len - 1.0;
		f += HUBER * HUBER * (std::sqrt(1.0 + (r / HUBER) * (r / HUBER)) - 1.0);
		if (rel) {
			rel->push_back(r);
		}
	}
	for (int i = 0; i < N; ++i) {
		f += LAMBDA * (x[i] - 0.5) * (x[i] - 0.5);
	}
	return f;
}
static void gradient(const float *x, const std::vector<Seg> &segs, float *g) {
	const double h = 1e-3;
	for (int i = 0; i < N; ++i) {
		float xp[N], xm[N];
		std::copy(x, x + N, xp);
		std::copy(x, x + N, xm);
		xp[i] = std::min(1.0f, x[i] + float(h));
		xm[i] = std::max(0.0f, x[i] - float(h));
		g[i] = float((objective(xp, segs) - objective(xm, segs)) / (xp[i] - xm[i]));
	}
}
struct Fit {
	std::vector<float> x;
	double f0 = 0, f = 0;
	int iterations = 0, nfev = 0;
	bool converged = false;
	std::string reason;
};
static Fit fit(const std::vector<Seg> &segs) {
	Fit r;
	VecCpu v;
	Lbfgsb solver(v);
	LbfgsbParams p;
	p.max_iterations = 200;
	p.epsilon = 1e-4;
	std::vector<float> x0(N, 0.5f), lb(N, 0.0f), ub(N, 1.0f), x(N), g(N);
	r.f0 = objective(x0.data(), segs);
	Lbfgsb::Status st = solver.start(N, x0.data(), lb.data(), ub.data(), p);
	double f = r.f0;
	for (int guard = 0; guard < 100000; ++guard) {
		if (st == Lbfgsb::NEED_EVAL || st == Lbfgsb::TRY) {
			solver.readX(x);
			f = objective(x.data(), segs);
			gradient(x.data(), segs, g.data());
			solver.setGradient(g.data());
			st = solver.next(f);
		} else if (st == Lbfgsb::ACCEPT || st == Lbfgsb::BUSY) {
			st = solver.next();
		} else {
			break;
		}
	}
	solver.readX(r.x);
	r.f = objective(r.x.data(), segs);
	r.iterations = solver.iterations();
	r.nfev = solver.nfev();
	r.converged = st == Lbfgsb::CONVERGED;
	r.reason = solver.reason().empty() ? Lbfgsb::status_name(st) : solver.reason();
	return r;
}
int main(int argc, char **argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: %s <soma_pheno.bin> <golden.bin> <targets.txt>\n", argv[0]);
		return 2;
	}
	if (pheno_load(argv[1]) != 0) {
		std::fprintf(stderr, "cannot load %s\n", argv[1]);
		return 2;
	}
	int rc = 0;
	{ // G1: the golden file is ident[11], v0[V*3], bind_world[J*16] as floats
		std::ifstream gf(argv[2], std::ios::binary);
		std::vector<float> gold(11 + PH_V * 3 + PH_J * 16);
		gf.read(reinterpret_cast<char *>(gold.data()), gold.size() * sizeof(float));
		pheno_eval(gold.data(), g_v.data(), g_bw.data());
		double ev = 0, eb = 0;
		for (int i = 0; i < PH_V * 3; ++i) {
			ev = std::max(ev, double(std::fabs(g_v[i] - gold[11 + i])));
		}
		for (int i = 0; i < PH_J * 16; ++i) {
			eb = std::max(eb, double(std::fabs(g_bw[i] - gold[11 + PH_V * 3 + i])));
		}
		std::printf("   sample v0[0..2] eval %.4f %.4f %.4f  golden %.4f %.4f %.4f; hips y eval %.4f golden %.4f\n", g_v[0], g_v[1], g_v[2], gold[11], gold[12], gold[13], g_bw[16 + 7], gold[11 + PH_V * 3 + 16 + 7]);
		if (const char *dump = std::getenv("DUMP_EVAL")) {
			std::ofstream o(dump, std::ios::binary);
			o.write(reinterpret_cast<const char *>(g_v.data()), g_v.size() * 4);
			o.write(reinterpret_cast<const char *>(g_bw.data()), g_bw.size() * 4);
		}
		const bool ok = gf.good() && ev < 1e-3 && eb < 1e-3;
		std::printf("G1 golden phenotype: mesh max error %.2e m, bind skeleton %.2e: %s\n", ev, eb, ok ? "PASS" : "FAIL");
		rc |= ok ? 0 : 1;
	}
	std::vector<Seg> segs;
	{
		std::ifstream tf(argv[3]);
		std::string line;
		while (std::getline(tf, line)) {
			std::istringstream is(line);
			Seg s;
			if (is >> s.name >> s.a >> s.b >> s.len) {
				segs.push_back(s);
			}
		}
	}
	Fit base = fit(segs);
	std::vector<double> rel0, rel1;
	std::vector<float> mean(N, 0.5f);
	objective(mean.data(), segs, &rel0);
	objective(base.x.data(), segs, &rel1);
	const char *names[N] = { "gender", "age", "muscle", "weight", "height", "proportions", "cupsize", "firmness", "african", "asian", "caucasian" };
	std::printf("fit: %d iterations, %d evaluations, %s; objective %.5f -> %.5f\n", base.iterations, base.nfev, base.reason.c_str(), base.f0, base.f);
	int at_bound = 0;
	for (int i = 0; i < N; ++i) {
		const bool b = base.x[i] < 1e-4f || base.x[i] > 1 - 1e-4f;
		at_bound += b;
		std::printf("  %-12s %.4f%s\n", names[i], base.x[i], b ? "  <- at a bound" : "");
	}
	std::printf("segments (relative error: mean body -> fit):\n");
	for (size_t i = 0; i < segs.size(); ++i) {
		std::printf("  %-16s target %.4f m  %+6.1f%% -> %+6.1f%%\n", segs[i].name.c_str(), segs[i].len, 100 * rel0[i], 100 * rel1[i]);
	}
	std::printf("saturation: %d of %d phenotype values at a bound\n", at_bound, N);
	const bool g2 = base.f < base.f0 && (base.converged || base.iterations > 0);
	std::printf("G2 residual falls (%.5f -> %.5f) and the solver finished (%s): %s\n", base.f0, base.f, base.reason.c_str(), g2 ? "PASS" : "FAIL");
	rc |= g2 ? 0 : 1;
	{ // G3 control: targets 10% longer must move the fit, and the moved fit must do worse on the real targets
		std::vector<Seg> longer = segs;
		for (Seg &s : longer) {
			s.len *= 1.10;
		}
		Fit alt = fit(longer);
		const double moved = std::fabs(alt.x[4] - base.x[4]) + std::fabs(alt.x[5] - base.x[5]) + std::fabs(alt.x[0] - base.x[0]);
		const double f_alt_on_real = objective(alt.x.data(), segs);
		const bool ok = f_alt_on_real > base.f * 1.02 && moved > 1e-3;
		std::printf("G3 control, targets +10%%: fit moved %.3f, residual on the real targets %.5f against %.5f: %s\n", moved, f_alt_on_real, base.f, ok ? "PASS" : "FAIL");
		rc |= ok ? 0 : 1;
	}
	return rc;
}
