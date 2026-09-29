# Head-fit oracle (Gate 11; host-native)

`gen.cpp` restates round 0 of the head fit (RFD 2275's method, `guest/headfit`)
in double, term for term as the Lean kernels define it, and solves it with
unmodified LBFGSpp 0.3.0:

    c_t = prod_{s in slots(t)} w_s(phen)      macro rows (anny_coeffs.gd's anchor rule)
    c_t = x[16 + k]                           dial rows (a positive and a negative per dial)
    v_i = base_i + sum_t c_t blend_ti                                  hf_blend
    p_i = s R(rot6) v_i + t                                            hf_similarity
    E   = sum_i wp_i (n_i . (p_i - q_i))^2 + wq_i |p_i - q_i|^2        hf_surface_residual
        + sum_k lw_k |p_mid_k - tgt_k|^2                               hf_landmark_residual
        + sum_j lam_j (x_j - x0_j)^2                                   hf_prior

`R` is the kernel's Gram-Schmidt of the two rot6 rows (norms floored at
1e-20); its derivative comes from one forward-mode dual pass per rot6 entry.
An axis value on an anchor takes the left segment, as `hf_core.cpp`'s
`interp` does.

Input: the dump `build/headfit_native/hf_native dump ... out=<dir>` writes
(tests/headfit): `meta.txt` and raw float32/uint32 arrays of the round's
correspondences, weights, marks, bounds, the model's base and blend rows, the
guest's round-0 optimum `xg` and its own loss and gradient at `x0`.

Checks (exit non-zero on any FAIL):
1. **f0**: the double loss at x0 against the guest's (relative, tol 1e-5);
   the analytic double gradient is first checked against central differences
   in double (one-sided on an anchor; tol 1e-5 of the largest component).
2. **gradcheck**: the guest's gradient (float32 kernels, df32 sums) against
   the double one, per block (rot6, t, s, phenotype, dials), as max error over
   the block's largest component, tol 1e-3: float32 per-vertex terms summed
   over ~1,100 vertices leave errors near 1e-5 of the block, so 1e-3 is
   loose enough for rounding and tight enough for a wrong term (the plant
   below is a 25% error).
3. **solve**: LBFGSpp's `LBFGSBSolver<double>` from x0 with the guest's bounds
   and parameters (m 6, epsilon 1e-4, epsilon_rel 0, past 1, delta 1e-8, the
   iteration cap from the dump, max_linesearch 30); the guest's optimum must
   reach the oracle's loss within 1e-3 relative. A diagnostic (not gated)
   caps LBFGSpp at the guest's iteration count.

Plants, which must fail (the gate passes only when they do):
- `--plant grad`: the guest's phenotype gradient scaled by 1.25;
- `--plant objective`: the landmark term dropped from the oracle.

## Run

```sh
build/headfit_native/hf_native dump anny=... dials=... map=... fixture=... marks=... out=<dir>
DUMP=<dir> tests/headfit_oracle/build.sh                # writes gates/11-headfit/oracle/
ARIA=<clone at 10086b6b> DUMP=<dir> tests/headfit_oracle/build.sh
```

Logs: `gen.log` (the real run), `plant_grad.log`, `plant_objective.log`.
The dump is generated data and is not committed; the logs report its
measurements.

`meta.txt`'s `tscale` and `pscale` give the units of x's translation (mm per
unit) and phenotype (per unit); the oracle applies them as the guest does.
`--trace` prints LBFGSpp's f after each iteration cap (a diagnostic).

## Result (2026-09-29, the Mire dump; P 4422, NB 530, n 158): FAIL on the solve

- f0: double 164.605031182, guest 164.604991528, relative 2.4e-7: PASS.
- Analytic vs central differences in double: 1.4e-9: PASS.
- gradcheck (tol 1e-3): rot6 1.0e-4, t 2.4e-5, s 3.1e-5, phenotype 2.9e-5,
  dials 1.4e-5: PASS.
- solve: guest round 0 f 68.4095 (68.4095 in double) against LBFGSpp's
  68.5897, both at the 60-iteration cap: −2.6e-3, outside the two-sided 1e-3:
  FAIL. The guest ends lower. The x blocks agree within 0.0025 (rot6), 0.023
  (t), 0.00046 (s), 0.0094 (phenotype) and 0.044 (dials).
- Plants: `grad` fails the gradcheck, `objective` fails f0 and the solve.

An earlier run, on another avatar, failed the solve the other way, and that
failure is why the guest changed. The guest stopped round 0 after 21 iterations (f 130.46) against LBFGSpp's
115.99. `--trace` showed the paths together to iteration ~30, then LBFGSpp
leaving a plateau the float32 guest stayed on. The causes were the
translation held at 1,141 mm (float32 ulp 1.2e-4 mm: a step could leave x
unchanged) and unknowns scaled 40:1 (phenotype against dials), which spoiled
float32 curvature pairs. The avatar mesh is now centred in the core, and x
carries the translation in 100 mm and the phenotype in 0.1, with the same
objective.
