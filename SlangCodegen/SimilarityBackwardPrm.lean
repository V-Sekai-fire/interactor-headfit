import Drape.SlangCodegen.Dsl
import Drape.SlangCodegen.Common
import Anny.SlangCodegen.Common

/-!
# `HeadFit.SlangCodegen.SimilarityBackwardPrm` — the VJP of `hf_similarity`

Two kernels over the cotangent `dpos` of `pos = s·R·v + t`:

* `hf_similarity_backward_v` (the sibling module), one thread per vertex:

      dv[i] = s · Rᵀ · dpos[i]

* this kernel (`hf_similarity_backward_prm`), one thread in all: the
  thirteen sums over the vertices, each in df32 in vertex order,

      dt = Σ dpos[i],   ds = Σ dpos[i] · (R v[i]),   dR = s · Σ dpos[i] v[i]ᵀ

  then `dR` through the 6D map's VJP (`Anny.SlangCodegen.Common.rot6Backward`)
  into `dprm[0 … 5]`, with `dprm[6 … 8] = dt` and `dprm[9] = ds`.
  A serial kernel (the `*_serial` shape): it has a cpp emit and runs the
  same sums in the same order on both targets.

Bindings (set 0), `hf_similarity_backward_v`:

  0  ConstantBuffer<HfSimParams> { uint P; }
  1  StructuredBuffer<float>   prm   (10)
  2  StructuredBuffer<float>   dpos  (P·3)
  3  RWStructuredBuffer<float> dv    (P·3)

This kernel:

  0  ConstantBuffer<HfSimParams> { uint P; }
  1  StructuredBuffer<float>   prm   (10)
  2  StructuredBuffer<float>   v     (P·3)
  3  StructuredBuffer<float>   dpos  (P·3)
  4  RWStructuredBuffer<float> dprm  (10)
-/

namespace HeadFit.SlangCodegen.SimilarityBackwardPrm

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

/-- The df32 accumulator pair `<n>_hi`, `<n>_lo`, zeroed. -/
private def acc (n : String) : List St := [ let_ fT (n ++ "_hi") (fl 0.0), let_ fT (n ++ "_lo") (fl 0.0) ]
/-- `(n_hi, n_lo) += a·b`. -/
private def accAdd (n : String) (a b : E) : St := do_ (call "df_acc" [v (n ++ "_hi"), v (n ++ "_lo"), a, b])
private def accVal (n : String) : E := v (n ++ "_hi") + v (n ++ "_lo")

def shader : SlangShaderModule :=
  { structs := [ { name := "HfSimParams", fields := [fld "P" uT] } ]
  , globals := [ paramsCB "HfSimParams", roF "prm" 1, roF "v" 2, roF "dpos" 3, rwF "dprm" 4 ]
  , functions := Drape.SlangCodegen.Common.dfHelpers ++
      [ entry 1 [dtid]
          ([ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ] ] ++
           rot6 "R" "prm" (u 0) ++
           (r9.flatMap fun k => acc ("m" ++ toString k)) ++
           (r3.flatMap fun k => acc ("t" ++ toString k)) ++
           acc "sd" ++
           [ for_ "i" (u 0) (p "P")
               (vec "g" (fun k => at_ "dpos" (v "i" * u 3 + u k)) ++
                vec "x" (fun k => at_ "v" (v "i" * u 3 + u k)) ++
                matVec "rx" "R" "x" ++
                (r9.map fun k => accAdd ("m" ++ toString k) (s "g" (k / 3)) (s "x" (k % 3))) ++
                (r3.map fun k => accAdd ("t" ++ toString k) (s "g" k) (fl 1.0)) ++
                (r3.map fun k => accAdd "sd" (s "g" k) (s "rx" k))) ] ++
           [ let_ fT "sc" (at_ "prm" (u 9)) ] ++
           mat "dR" (fun i j => v "sc" * accVal ("m" ++ toString (3 * i + j))) ++
           rot6Backward "R" "dR" "dprm" (u 0) ++
           (r3.map fun k => setAt "dprm" (u (6 + k)) (accVal ("t" ++ toString k))) ++
           [ setAt "dprm" (u 9) (accVal "sd") ]) ] }

-- BEGIN PIN
def expected : String :=
"struct HfSimParams {
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfSimParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> prm;
[[vk::binding(2, 0)]]
StructuredBuffer<float> v;
[[vk::binding(3, 0)]]
StructuredBuffer<float> dpos;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dprm;

void two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float bb = (h - a);
  float ah = (h - bb);
  float lo_a = (a - ah);
  float lo_b = (b - bb);
  hi = h;
  lo = (lo_a + lo_b);
  return;
}

void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = (a + b);
  float t = (h - a);
  hi = h;
  lo = (b - t);
  return;
}

void two_prod(float a, float b, out float hi, out float lo) {
  float h = (a * b);
  hi = h;
  lo = fma(a, b, (-h));
  return;
}

void df_add(float x_hi, float x_lo, float y_hi, float y_lo, out float z_hi, out float z_lo) {
  float sh;
  float sl;
  two_sum(x_hi, y_hi, sh, sl);
  float xy_lo = (x_lo + y_lo);
  float sl2 = (sl + xy_lo);
  quick_two_sum(sh, sl2, z_hi, z_lo);
  return;
}

void df_acc(inout float hi, inout float lo, float a, float b) {
  float p_hi;
  float p_lo;
  two_prod(a, b, p_hi, p_lo);
  float n_hi;
  float n_lo;
  df_add(hi, lo, p_hi, p_lo, n_hi, n_lo);
  hi = n_hi;
  lo = n_lo;
  return;
}

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  float R_a10 = prm[(0u + 0u)];
  float R_a11 = prm[(0u + 1u)];
  float R_a12 = prm[(0u + 2u)];
  float R_a20 = prm[(0u + 3u)];
  float R_a21 = prm[(0u + 4u)];
  float R_a22 = prm[(0u + 5u)];
  float R_n1 = max(sqrt((((R_a10 * R_a10) + (R_a11 * R_a11)) + (R_a12 * R_a12))), 1.0e-20f);
  float R_r10 = (R_a10 / R_n1);
  float R_r11 = (R_a11 / R_n1);
  float R_r12 = (R_a12 / R_n1);
  float R_d = (((R_r10 * R_a20) + (R_r11 * R_a21)) + (R_r12 * R_a22));
  float R_u0 = (R_a20 - (R_d * R_r10));
  float R_u1 = (R_a21 - (R_d * R_r11));
  float R_u2 = (R_a22 - (R_d * R_r12));
  float R_nu = max(sqrt((((R_u0 * R_u0) + (R_u1 * R_u1)) + (R_u2 * R_u2))), 1.0e-20f);
  float R_r20 = (R_u0 / R_nu);
  float R_r21 = (R_u1 / R_nu);
  float R_r22 = (R_u2 / R_nu);
  float R_r30 = ((R_r11 * R_r22) - (R_r12 * R_r21));
  float R_r31 = ((R_r12 * R_r20) - (R_r10 * R_r22));
  float R_r32 = ((R_r10 * R_r21) - (R_r11 * R_r20));
  float R0 = R_r10;
  float R1 = R_r11;
  float R2 = R_r12;
  float R3 = R_r20;
  float R4 = R_r21;
  float R5 = R_r22;
  float R6 = R_r30;
  float R7 = R_r31;
  float R8 = R_r32;
  float m0_hi = 0.000000;
  float m0_lo = 0.000000;
  float m1_hi = 0.000000;
  float m1_lo = 0.000000;
  float m2_hi = 0.000000;
  float m2_lo = 0.000000;
  float m3_hi = 0.000000;
  float m3_lo = 0.000000;
  float m4_hi = 0.000000;
  float m4_lo = 0.000000;
  float m5_hi = 0.000000;
  float m5_lo = 0.000000;
  float m6_hi = 0.000000;
  float m6_lo = 0.000000;
  float m7_hi = 0.000000;
  float m7_lo = 0.000000;
  float m8_hi = 0.000000;
  float m8_lo = 0.000000;
  float t0_hi = 0.000000;
  float t0_lo = 0.000000;
  float t1_hi = 0.000000;
  float t1_lo = 0.000000;
  float t2_hi = 0.000000;
  float t2_lo = 0.000000;
  float sd_hi = 0.000000;
  float sd_lo = 0.000000;
  for (uint i = 0u; i < params.P; ++i) {
    float g0 = dpos[((i * 3u) + 0u)];
    float g1 = dpos[((i * 3u) + 1u)];
    float g2 = dpos[((i * 3u) + 2u)];
    float x0 = v[((i * 3u) + 0u)];
    float x1 = v[((i * 3u) + 1u)];
    float x2 = v[((i * 3u) + 2u)];
    float rx0 = (((R0 * x0) + (R1 * x1)) + (R2 * x2));
    float rx1 = (((R3 * x0) + (R4 * x1)) + (R5 * x2));
    float rx2 = (((R6 * x0) + (R7 * x1)) + (R8 * x2));
    df_acc(m0_hi, m0_lo, g0, x0);
    df_acc(m1_hi, m1_lo, g0, x1);
    df_acc(m2_hi, m2_lo, g0, x2);
    df_acc(m3_hi, m3_lo, g1, x0);
    df_acc(m4_hi, m4_lo, g1, x1);
    df_acc(m5_hi, m5_lo, g1, x2);
    df_acc(m6_hi, m6_lo, g2, x0);
    df_acc(m7_hi, m7_lo, g2, x1);
    df_acc(m8_hi, m8_lo, g2, x2);
    df_acc(t0_hi, t0_lo, g0, 1.000000);
    df_acc(t1_hi, t1_lo, g1, 1.000000);
    df_acc(t2_hi, t2_lo, g2, 1.000000);
    df_acc(sd_hi, sd_lo, g0, rx0);
    df_acc(sd_hi, sd_lo, g1, rx1);
    df_acc(sd_hi, sd_lo, g2, rx2);
  }
  float sc = prm[9u];
  float dR0 = (sc * (m0_hi + m0_lo));
  float dR1 = (sc * (m1_hi + m1_lo));
  float dR2 = (sc * (m2_hi + m2_lo));
  float dR3 = (sc * (m3_hi + m3_lo));
  float dR4 = (sc * (m4_hi + m4_lo));
  float dR5 = (sc * (m5_hi + m5_lo));
  float dR6 = (sc * (m6_hi + m6_lo));
  float dR7 = (sc * (m7_hi + m7_lo));
  float dR8 = (sc * (m8_hi + m8_lo));
  float R_g10 = dR0;
  float R_g11 = dR1;
  float R_g12 = dR2;
  float R_g20 = dR3;
  float R_g21 = dR4;
  float R_g22 = dR5;
  float R_g30 = dR6;
  float R_g31 = dR7;
  float R_g32 = dR8;
  float R_c10 = ((R_r21 * R_g32) - (R_r22 * R_g31));
  float R_c11 = ((R_r22 * R_g30) - (R_r20 * R_g32));
  float R_c12 = ((R_r20 * R_g31) - (R_r21 * R_g30));
  float R_c20 = ((R_g31 * R_r12) - (R_g32 * R_r11));
  float R_c21 = ((R_g32 * R_r10) - (R_g30 * R_r12));
  float R_c22 = ((R_g30 * R_r11) - (R_g31 * R_r10));
  float R_h10 = (R_g10 + R_c10);
  float R_h11 = (R_g11 + R_c11);
  float R_h12 = (R_g12 + R_c12);
  float R_h20 = (R_g20 + R_c20);
  float R_h21 = (R_g21 + R_c21);
  float R_h22 = (R_g22 + R_c22);
  float R_p2 = (((R_r20 * R_h20) + (R_r21 * R_h21)) + (R_r22 * R_h22));
  float R_du0 = ((R_h20 - (R_p2 * R_r20)) / R_nu);
  float R_du1 = ((R_h21 - (R_p2 * R_r21)) / R_nu);
  float R_du2 = ((R_h22 - (R_p2 * R_r22)) / R_nu);
  float R_q = (((R_r10 * R_du0) + (R_r11 * R_du1)) + (R_r12 * R_du2));
  float R_da20 = (R_du0 - (R_q * R_r10));
  float R_da21 = (R_du1 - (R_q * R_r11));
  float R_da22 = (R_du2 - (R_q * R_r12));
  float R_h1b0 = (R_h10 - ((R_d * R_du0) + (R_q * R_a20)));
  float R_h1b1 = (R_h11 - ((R_d * R_du1) + (R_q * R_a21)));
  float R_h1b2 = (R_h12 - ((R_d * R_du2) + (R_q * R_a22)));
  float R_p1 = (((R_r10 * R_h1b0) + (R_r11 * R_h1b1)) + (R_r12 * R_h1b2));
  float R_da10 = ((R_h1b0 - (R_p1 * R_r10)) / R_n1);
  float R_da11 = ((R_h1b1 - (R_p1 * R_r11)) / R_n1);
  float R_da12 = ((R_h1b2 - (R_p1 * R_r12)) / R_n1);
  dprm[(0u + 0u)] = R_da10;
  dprm[(0u + 1u)] = R_da11;
  dprm[(0u + 2u)] = R_da12;
  dprm[(0u + 3u)] = R_da20;
  dprm[(0u + 4u)] = R_da21;
  dprm[(0u + 5u)] = R_da22;
  dprm[6u] = (t0_hi + t0_lo);
  dprm[7u] = (t1_hi + t1_lo);
  dprm[8u] = (t2_hi + t2_lo);
  dprm[9u] = (sd_hi + sd_lo);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.SimilarityBackwardPrm
