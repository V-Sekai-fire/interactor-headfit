// Head-fit oracle (Gate 11): host-native, double throughout. Restates round 0
// of RFD 2275's head fit, as headfit's Lean kernels define it, and solves it
// with unmodified LBFGSpp 0.3.0, so the guest's float32/df32 pipeline and its
// L-BFGS-B port are checked against an independent solve of the same problem.
//
//   c_t   = prod_{s in slots(t)} w_s(phen)       macro row t (anny_coeffs.gd's rule)
//   c_t   = x[16 + k]                            dial row k (pos, neg per dial)
//   v_i   = base_i + sum_t c_t blend_ti          (hf_blend)
//   p_i   = s R(rot6) v_i + t                    (hf_similarity: Gram-Schmidt rows)
//   E     = sum_i wp_i (n_i.(p_i - q_i))^2 + wq_i |p_i - q_i|^2     (hf_surface_residual)
//         + sum_k lw_k |p_mid_k - tgt_k|^2                           (hf_landmark_residual)
//         + sum_j lam_j (x_j - x0_j)^2                               (hf_prior)
//
// The input is the dump `hf_native dump` writes (tests/headfit): the round's
// correspondences, weights, marks and bounds after the landmark start, the
// model's base and blend rows, the guest's round-0 optimum and its own loss
// and gradient at x0.
//
// Checks: f0 (the loss at x0), the gradient at x0 (the double analytic one,
// itself checked against central differences, against the guest's), and the
// solve (LBFGSpp from x0 with the guest's parameters against the guest's
// optimum). Plants that must be caught: `--plant grad` scales the guest's
// phenotype gradient by 1.25; `--plant objective` drops the landmark term.
//
// Usage: gen <dumpdir> [--plant grad|objective]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#define private public
#include <LBFGSB.h>
#undef private

using Vec = Eigen::VectorXd;

namespace {

struct Prob {
    uint32_t P = 0, NB = 0, K = 0, n = 0, L = 0, macro = 0;
    int max_iterations = 60;
    double guest_f0 = 0, guest_f = 0;
    double tscale = 1.0; // mm per unit of x[6..8] (meta `tscale`)
    double pscale = 1.0; // phenotype per unit of x[10..15] (meta `pscale`)
    int guest_iters = 0;
    std::vector<std::vector<int>> axis_slots;
    std::vector<std::vector<double>> anchors;
    std::vector<std::vector<int>> row_slots;
    std::vector<float> base, blend, q, nrm, w, tgt, lw, x0, lb, ub, px0, plam, xg, g0;
    std::vector<uint32_t> mid;
    bool no_landmarks = false; // the objective plant
};

template <class T>
bool read_bin(const std::string& path, std::vector<T>& v, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    v.resize(n);
    const size_t got = std::fread(v.data(), sizeof(T), n, f);
    char extra;
    const bool tail = std::fread(&extra, 1, 1, f) == 1;
    std::fclose(f);
    return got == n && !tail;
}

bool load(const std::string& dir, Prob& P) {
    std::ifstream is(dir + "/meta.txt");
    if (!is) return false;
    std::string line;
    while (std::getline(is, line)) {
        std::istringstream ls(line);
        std::string kw;
        ls >> kw;
        if (kw == "P") {
            std::string k;
            ls >> P.P >> k >> P.NB >> k >> P.K >> k >> P.n >> k >> P.L >> k >> P.macro;
        } else if (kw == "lbfgsb") {
            std::string k;
            while (ls >> k) if (k == "max_iterations") ls >> P.max_iterations;
        } else if (kw == "tscale") {
            ls >> P.tscale;
        } else if (kw == "pscale") {
            ls >> P.pscale;
        } else if (kw == "guest") {
            std::string k;
            ls >> k >> P.guest_f0 >> k >> P.guest_iters >> k >> P.guest_f;
        } else if (kw == "axis") {
            std::vector<int> s;
            std::vector<double> a;
            std::string t;
            bool anc = false;
            while (ls >> t) {
                if (t == "anchors") { anc = true; continue; }
                if (anc) a.push_back(std::strtod(t.c_str(), nullptr)); else s.push_back(std::atoi(t.c_str()));
            }
            P.axis_slots.push_back(s);
            P.anchors.push_back(a);
        } else if (kw == "row") {
            std::vector<int> s;
            int v;
            while (ls >> v) s.push_back(v);
            P.row_slots.push_back(s);
        }
    }
    const size_t P3 = size_t(P.P) * 3;
    return P.axis_slots.size() == 6 && P.row_slots.size() == P.macro && P.NB == P.macro + 2 * P.L &&
           P.n == 16 + 2 * P.L && read_bin(dir + "/base.f32", P.base, P3) &&
           read_bin(dir + "/blend.f32", P.blend, size_t(P.NB) * P3) && read_bin(dir + "/q.f32", P.q, P3) &&
           read_bin(dir + "/nrm.f32", P.nrm, P3) && read_bin(dir + "/w.f32", P.w, size_t(P.P) * 2) &&
           read_bin(dir + "/mid.u32", P.mid, P.K) && read_bin(dir + "/tgt.f32", P.tgt, size_t(P.K) * 3) &&
           read_bin(dir + "/lw.f32", P.lw, P.K) && read_bin(dir + "/x0.f32", P.x0, P.n) &&
           read_bin(dir + "/lb.f32", P.lb, P.n) && read_bin(dir + "/ub.f32", P.ub, P.n) &&
           read_bin(dir + "/px0.f32", P.px0, P.n) && read_bin(dir + "/plam.f32", P.plam, P.n) &&
           read_bin(dir + "/xg.f32", P.xg, P.n) && read_bin(dir + "/g0.f32", P.g0, P.n);
}

// A forward-mode dual number: the rotation's derivative in each rot6 entry.
struct D {
    double v, d;
};
D operator+(D a, D b) { return { a.v + b.v, a.d + b.d }; }
D operator-(D a, D b) { return { a.v - b.v, a.d - b.d }; }
D operator*(D a, D b) { return { a.v * b.v, a.d * b.v + a.v * b.d }; }
D operator/(D a, D b) { return { a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v) }; }
D dsqrt(D a) { const double s = std::sqrt(a.v); return { s, a.d / (2 * s) }; }
D dmax(D a, double m) { return a.v > m ? a : D{ m, 0.0 }; }

// hf_similarity's rotation: rows r1 = a1/|a1|, r2 = GS(a2), r3 = r1 x r2
// (the norms floored at 1e-20 as the kernel does), row-major.
void rot(const D a[6], D R[9]) {
    const D n1 = dmax(dsqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), 1e-20);
    const D r1[3] = { a[0] / n1, a[1] / n1, a[2] / n1 };
    const D d = r1[0] * a[3] + r1[1] * a[4] + r1[2] * a[5];
    const D u[3] = { a[3] - d * r1[0], a[4] - d * r1[1], a[5] - d * r1[2] };
    const D nu = dmax(dsqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), 1e-20);
    const D r2[3] = { u[0] / nu, u[1] / nu, u[2] / nu };
    const D r3[3] = { r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0] };
    for (int k = 0; k < 3; ++k) R[k] = r1[k], R[3 + k] = r2[k], R[6 + k] = r3[k];
}

// hf_core.cpp interp(): the anchor weights of one axis and their derivative
// (the segment is the first whose upper anchor is not below the value, so an
// anchor value takes the left segment; outside the clamp the derivative is 0).
void interp(double value, const std::vector<double>& an, std::vector<double>& w, std::vector<double>& dw) {
    const int n = int(an.size());
    w.assign(n, 0.0), dw.assign(n, 0.0);
    int idx = 1;
    while (idx < n - 1 && an[idx] < value) ++idx;
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

struct Terms {
    double surface = 0, landmark = 0, prior = 0;
    double total() const { return surface + landmark + prior; }
};

// The loss and its analytic gradient, in double.
double eval(const Prob& P, const Vec& x, Vec& g, Terms* terms = nullptr) {
    const uint32_t Pn = P.P;
    double sv[17] = { 0 }, dsv[17] = { 0 };
    int slot_axis[17] = { 0 };
    for (int a = 0; a < 6; ++a) {
        std::vector<double> w, dw;
        interp(P.pscale * x[10 + a], P.anchors[a], w, dw);
        for (size_t k = 0; k < P.axis_slots[a].size(); ++k) {
            const int s = P.axis_slots[a][k];
            sv[s] = w[k], dsv[s] = dw[k], slot_axis[s] = a;
        }
    }
    std::vector<double> c(P.NB, 0.0);
    for (uint32_t t = 0; t < P.macro; ++t) {
        double w = 1.0;
        for (int s : P.row_slots[t]) w *= sv[s];
        c[t] = w;
    }
    for (uint32_t k = 0; k < 2 * P.L; ++k) c[P.macro + k] = x[16 + k];
    std::vector<double> v(size_t(Pn) * 3);
    for (size_t k = 0; k < v.size(); ++k) v[k] = P.base[k];
    for (uint32_t t = 0; t < P.NB; ++t) {
        if (c[t] == 0.0) continue;
        const float* b = &P.blend[size_t(t) * Pn * 3];
        for (size_t k = 0; k < v.size(); ++k) v[k] += c[t] * double(b[k]);
    }
    // R and dR/da_j by one dual pass per rot6 entry.
    double R[9], dR[6][9];
    for (int j = 0; j < 6; ++j) {
        D a[6], Rd[9];
        for (int k = 0; k < 6; ++k) a[k] = { x[k], k == j ? 1.0 : 0.0 };
        rot(a, Rd);
        for (int k = 0; k < 9; ++k) R[k] = Rd[k].v, dR[j][k] = Rd[k].d;
    }
    const double s = x[9];
    std::vector<double> pos(v.size()), rv(v.size()), dpos(v.size(), 0.0);
    for (uint32_t i = 0; i < Pn; ++i)
        for (int r = 0; r < 3; ++r) {
            rv[i * 3 + r] = R[r * 3] * v[i * 3] + R[r * 3 + 1] * v[i * 3 + 1] + R[r * 3 + 2] * v[i * 3 + 2];
            pos[i * 3 + r] = s * rv[i * 3 + r] + P.tscale * x[6 + r];
        }
    Terms T;
    for (uint32_t i = 0; i < Pn; ++i) {
        const double wp = P.w[i * 2], wq = P.w[i * 2 + 1];
        if (wp == 0.0 && wq == 0.0) continue;
        double d[3], pl = 0, dd = 0;
        for (int r = 0; r < 3; ++r) {
            d[r] = pos[i * 3 + r] - P.q[i * 3 + r];
            pl += P.nrm[i * 3 + r] * d[r];
            dd += d[r] * d[r];
        }
        T.surface += wp * pl * pl + wq * dd;
        for (int r = 0; r < 3; ++r) dpos[i * 3 + r] += 2 * wp * pl * P.nrm[i * 3 + r] + 2 * wq * d[r];
    }
    if (!P.no_landmarks)
        for (uint32_t k = 0; k < P.K; ++k) {
            const size_t o = size_t(P.mid[k]) * 3;
            double dd = 0;
            for (int r = 0; r < 3; ++r) {
                const double d = pos[o + r] - P.tgt[k * 3 + r];
                dd += d * d;
                dpos[o + r] += 2 * P.lw[k] * d;
            }
            T.landmark += P.lw[k] * dd;
        }
    g = Vec::Zero(P.n);
    for (uint32_t j = 0; j < P.n; ++j) {
        const double d = x[j] - P.px0[j];
        T.prior += P.plam[j] * d * d;
        g[j] += 2 * P.plam[j] * d;
    }
    // Similarity: t, s, and dE/dR = s sum_i dpos_i v_i^T through dR/da.
    double gR[9] = { 0 };
    for (uint32_t i = 0; i < Pn; ++i)
        for (int r = 0; r < 3; ++r) {
            const double dp = dpos[i * 3 + r];
            g[6 + r] += P.tscale * dp;
            g[9] += dp * rv[i * 3 + r];
            for (int k = 0; k < 3; ++k) gR[r * 3 + k] += s * dp * v[i * 3 + k];
        }
    for (int j = 0; j < 6; ++j)
        for (int k = 0; k < 9; ++k) g[j] += gR[k] * dR[j][k];
    // dv = s R^T dpos; dc_t = blend_t . dv.
    std::vector<double> dv(v.size());
    for (uint32_t i = 0; i < Pn; ++i)
        for (int k = 0; k < 3; ++k)
            dv[i * 3 + k] = s * (R[k] * dpos[i * 3] + R[3 + k] * dpos[i * 3 + 1] + R[6 + k] * dpos[i * 3 + 2]);
    std::vector<double> dc(P.NB, 0.0);
    for (uint32_t t = 0; t < P.NB; ++t) {
        const float* b = &P.blend[size_t(t) * Pn * 3];
        double acc = 0;
        for (size_t k = 0; k < dv.size(); ++k) acc += double(b[k]) * dv[k];
        dc[t] = acc;
    }
    for (uint32_t t = 0; t < P.macro; ++t) {
        const std::vector<int>& sl = P.row_slots[t];
        for (size_t k = 0; k < sl.size(); ++k) {
            double prod = dsv[sl[k]];
            for (size_t j = 0; j < sl.size(); ++j)
                if (j != k) prod *= sv[sl[j]];
            g[10 + slot_axis[sl[k]]] += dc[t] * prod * P.pscale;
        }
    }
    for (uint32_t k = 0; k < 2 * P.L; ++k) g[16 + k] += dc[P.macro + k];
    if (terms) *terms = T;
    return T.total();
}

Vec tovec(const std::vector<float>& v) {
    Vec r(v.size());
    for (size_t k = 0; k < v.size(); ++k) r[k] = v[k];
    return r;
}

const char* kBlock[5] = { "rot6", "t", "s", "phenotype", "dials" };
int block_of(int j) { return j < 6 ? 0 : j < 9 ? 1 : j < 10 ? 2 : j < 16 ? 3 : 4; }

// Per block: max |a - b| over the block, relative to max |b| in the block.
void block_err(const Vec& a, const Vec& b, double out[5], int worst[5]) {
    double num[5] = { 0 }, den[5] = { 0 };
    for (int k = 0; k < 5; ++k) worst[k] = -1;
    for (int j = 0; j < int(a.size()); ++j) {
        const int k = block_of(j);
        const double e = std::fabs(a[j] - b[j]);
        if (e > num[k]) num[k] = e, worst[k] = j;
        den[k] = std::max(den[k], std::fabs(b[j]));
    }
    for (int k = 0; k < 5; ++k) out[k] = den[k] > 0 ? num[k] / den[k] : num[k];
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: gen <dumpdir> [--plant grad|objective]\n");
        return 2;
    }
    std::string plant;
    for (int k = 2; k + 1 < argc; ++k)
        if (std::strcmp(argv[k], "--plant") == 0) plant = argv[k + 1];
    Prob P;
    if (!load(argv[1], P)) {
        std::printf("FAIL cannot load the dump %s\n", argv[1]);
        return 2;
    }
    if (plant == "objective") P.no_landmarks = true;
    std::printf("dump P %u NB %u K %u n %u L %u macro_rows %u; guest f0 %.9g, round-0 f %.9g in %d iterations\n",
                P.P, P.NB, P.K, P.n, P.L, P.macro, P.guest_f0, P.guest_f, P.guest_iters);
    if (!plant.empty()) std::printf("PLANT %s\n", plant.c_str());
    int fails = 0;
    auto verdict = [&](bool ok) { fails += ok ? 0 : 1; return ok ? "PASS" : "FAIL"; };

    // 1. The loss at x0.
    const Vec x0 = tovec(P.x0);
    Vec g(P.n);
    Terms T0;
    const double f0 = eval(P, x0, g, &T0);
    const double f0rel = std::fabs(f0 - P.guest_f0) / std::fabs(f0);
    const double f0tol = 1e-5;
    std::printf("f0: double %.12g (surface %.9g landmark %.9g prior %.9g) guest %.12g rel %.3e (tol %.0e) %s\n", f0,
                T0.surface, T0.landmark, T0.prior, P.guest_f0, f0rel, f0tol, verdict(f0rel <= f0tol));

    // The analytic double gradient against central differences in double.
    // Phenotype values on an anchor take the left segment (hf_core interp), so
    // there the difference is one-sided from the left.
    Vec fd(P.n), gt(P.n);
    double fdmax = 0, gmax = 0;
    int fdworst = -1;
    for (uint32_t j = 0; j < P.n; ++j) {
        const double h = 1e-6 * std::max(1.0, std::fabs(x0[j]));
        bool on_anchor = false;
        if (j >= 10 && j < 16)
            for (double a : P.anchors[j - 10]) on_anchor = on_anchor || P.pscale * x0[j] == a;
        Vec xp = x0, xm = x0;
        double fp, fm;
        if (on_anchor) {
            xm[j] -= h;
            fp = f0, fm = eval(P, xm, gt);
            fd[j] = (fp - fm) / h;
        } else {
            xp[j] += h, xm[j] -= h;
            fp = eval(P, xp, gt), fm = eval(P, xm, gt);
            fd[j] = (fp - fm) / (2 * h);
        }
        const double e = std::fabs(fd[j] - g[j]);
        if (e > fdmax) fdmax = e, fdworst = int(j);
        gmax = std::max(gmax, std::fabs(g[j]));
    }
    const double fdrel = fdmax / gmax, fdtol = 1e-5;
    std::printf("analytic vs central differences (double): worst |fd - g| / max|g| = %.3e at %d (tol %.0e) %s\n", fdrel,
                fdworst, fdtol, verdict(fdrel <= fdtol));

    // 2. The guest's gradient at x0 (float32 kernels, df32 sums).
    Vec gg = tovec(P.g0);
    if (plant == "grad")
        for (int a = 0; a < 6; ++a) gg[10 + a] *= 1.25;
    double be[5];
    int bw[5];
    block_err(gg, g, be, bw);
    const double gtol = 1e-3;
    bool gok = true;
    for (int k = 0; k < 5; ++k) {
        const bool ok = be[k] <= gtol;
        gok = gok && ok;
        std::printf("gradcheck %-9s max|guest - double| / max|double| = %.3e (worst %d: guest %.9g double %.9g) %s\n",
                    kBlock[k], be[k], bw[k], bw[k] >= 0 ? gg[bw[k]] : 0.0, bw[k] >= 0 ? g[bw[k]] : 0.0,
                    ok ? "ok" : "over");
    }
    std::printf("gradcheck (tol %.0e per block): %s\n", gtol, verdict(gok));

    // 3. The solve: LBFGSpp from x0 with the guest's parameters.
    LBFGSpp::LBFGSBParam<double> prm;
    prm.m = 6;
    prm.epsilon = 1e-4;
    prm.epsilon_rel = 0.0;
    prm.past = 1;
    prm.delta = 1e-8;
    prm.max_iterations = P.max_iterations;
    prm.max_linesearch = 30;
    LBFGSpp::LBFGSBSolver<double> solver(prm);
    Vec lb = tovec(P.lb), ub = tovec(P.ub);
    Vec xo = x0;
    double fo = 0;
    int evals = 0;
    std::function<double(const Vec&, Vec&)> fun = [&](const Vec& x, Vec& gr) { ++evals; return eval(P, x, gr); };
    int it = -1;
    std::string stop = "converged";
    try {
        it = solver.minimize(fun, xo, fo, lb, ub);
    } catch (const std::exception& e) {
        stop = e.what();
        Vec gr(P.n);
        fo = eval(P, xo, gr);
    }
    const Vec xg = tovec(P.xg);
    Vec ggx(P.n);
    const double fxg = eval(P, xg, ggx);
    const double srel = (fxg - fo) / fo, stol = 1e-3;
    std::printf("solve: LBFGSpp %d iterations, %d evaluations (%s): f %.9g -> %.9g; guest x* (%d iterations, its "
                "own f %.9g) at double f %.9g\n",
                it, evals, stop.c_str(), f0, fo, P.guest_iters, P.guest_f, fxg);
    for (int k = 0; k < 5; ++k) {
        double m = 0;
        for (uint32_t j = 0; j < P.n; ++j)
            if (block_of(int(j)) == k) m = std::max(m, std::fabs(xg[j] - xo[j]));
        std::printf("solve %-9s max|x_guest - x_oracle| = %.4g\n", kBlock[k], m);
    }
    std::printf("solve: (f(x_guest) - f_oracle) / f_oracle = %+.3e (tol |.| <= %.0e) %s\n", srel, stol,
                verdict(std::fabs(srel) <= stol));

    // --trace: LBFGSpp's f after each iteration count (a fresh solve capped at
    // k), to line up against the guest's own trace (hf_native verbose=1).
    for (int k = 1; k < argc; ++k)
        if (std::strcmp(argv[k], "--trace") == 0)
            for (int cap = 1; cap <= P.max_iterations; ++cap) {
                LBFGSpp::LBFGSBParam<double> p3 = prm;
                p3.max_iterations = cap;
                LBFGSpp::LBFGSBSolver<double> s3(p3);
                Vec xc = x0;
                double fc = 0;
                try {
                    s3.minimize(fun, xc, fc, lb, ub);
                } catch (const std::exception& e) {
                    std::printf("trace %d stopped: %s\n", cap, e.what());
                    break;
                }
                int at_lb = 0, at_ub = 0;
                for (uint32_t j = 0; j < P.n; ++j) at_lb += xc[j] <= lb[j], at_ub += xc[j] >= ub[j];
                std::printf("trace %d f %.9g at_lb %d at_ub %d\n", cap, fc, at_lb, at_ub);
            }

    // Diagnostic (not gated): LBFGSpp capped at the guest's iteration count, to
    // tell a different path from an earlier stop.
    {
        LBFGSpp::LBFGSBParam<double> p2 = prm;
        p2.max_iterations = std::max(1, P.guest_iters);
        LBFGSpp::LBFGSBSolver<double> s2(p2);
        Vec xc = x0;
        double fc = 0;
        int it2 = -1;
        std::string st2 = "stopped";
        try {
            it2 = s2.minimize(fun, xc, fc, lb, ub);
        } catch (const std::exception& e) {
            st2 = e.what();
            Vec gr(P.n);
            fc = eval(P, xc, gr);
        }
        double m = 0;
        for (uint32_t j = 0; j < P.n; ++j) m = std::max(m, std::fabs(xg[j] - xc[j]));
        std::printf("diagnostic: LBFGSpp capped at %d iterations (%d taken, %s): f %.9g; guest %.9g; max|x_guest - x| "
                    "= %.4g\n",
                    p2.max_iterations, it2, st2.c_str(), fc, fxg, m);
    }
    std::printf("%s: %d check(s) failed\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
