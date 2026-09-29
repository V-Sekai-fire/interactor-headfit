import Drape.SlangCodegen.Dsl
import Drape.SlangCodegen.Common

/-!
# `HeadFit.SlangCodegen.BlendBackward` — `dcoeffs = blendᵀ · dv`

The vector-Jacobian product of `hf_blend` with respect to `coeffs`: one
thread per row, a P·3-long dot product accumulated in df32 in vertex
order, so every target sums the same terms in the same order.

Bindings (set 0):

  0  ConstantBuffer<HfBlendParams> { uint P; uint NB; }
  1  StructuredBuffer<float>   blend    (NB·P·3)
  2  StructuredBuffer<float>   dv       (P·3)
  3  RWStructuredBuffer<float> dcoeffs  (NB)
-/

namespace HeadFit.SlangCodegen.BlendBackward

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "HfBlendParams", fields := [fld "P" uT, fld "NB" uT] } ]
  , globals := [ paramsCB "HfBlendParams", roF "blend" 1, roF "dv" 2, rwF "dcoeffs" 3 ]
  , functions := Drape.SlangCodegen.Common.dfHelpers ++
      [ entry 64 [dtid]
          [ let_ uT "c" (.member (v "tid") "x")
          , if_ (ge (v "c") (p "NB")) [ ret ]
          , let_ uT "n" (p "P" * u 3)
          , let_ uT "o" (v "c" * v "n")
          , let_ fT "hi" (fl 0.0)
          , let_ fT "lo" (fl 0.0)
          , for_ "i" (u 0) (v "n")
              [ do_ (call "df_acc" [v "hi", v "lo", at_ "blend" (v "o" + v "i"), at_ "dv" (v "i")]) ]
          , setAt "dcoeffs" (v "c") (v "hi" + v "lo") ] ] }

-- BEGIN PIN
def expected : String :=
"struct HfBlendParams {
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(2, 0)]]
StructuredBuffer<float> dv;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dcoeffs;

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

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint c = tid.x;
  if ((c >= params.NB)) {
    return;
  }
  uint n = (params.P * 3u);
  uint o = (c * n);
  float hi = 0.000000;
  float lo = 0.000000;
  for (uint i = 0u; i < n; ++i) {
    df_acc(hi, lo, blend[(o + i)], dv[i]);
  }
  dcoeffs[c] = (hi + lo);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.BlendBackward
