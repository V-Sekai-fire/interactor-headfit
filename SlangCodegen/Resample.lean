import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common

/-!
# `HeadFit.SlangCodegen.Resample` — carry a shape across, one thread per target vertex

The transfer. Target vertex `j` hit the fitted model's triangle `hit[j]`
at barycentrics `bary[j]` (the driver's ray cast along the target's
normal, with its normal test); a miss is `hit[j] = 0xFFFFFFFF` and stays
unmatched (zero). The model's deltas `dm` (in the model's own frame, as
ANNY's target files give them) are blended at that point and carried
through the fit's rotation and scale into the target's frame:

    out[j] = s · R · Σ_k bary[j][k] · dm[tri[hit[j]][k]]

Bindings (set 0):

  0  ConstantBuffer<HfResampleParams> { uint N; }   N target vertices
  1  StructuredBuffer<float>   prm   (10: rot6, t, s)
  2  StructuredBuffer<uint>    tri   (T·3, model region indices)
  3  StructuredBuffer<uint>    hit   (N)
  4  StructuredBuffer<float>   bary  (N·3)
  5  StructuredBuffer<float>   dm    (P·3)
  6  RWStructuredBuffer<float> out   (N·3)
-/

namespace HeadFit.SlangCodegen.Resample

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

def shader : SlangShaderModule :=
  { structs := [ { name := "HfResampleParams", fields := [fld "N" uT] } ]
  , globals :=
      [ paramsCB "HfResampleParams", roF "prm" 1, roU "tri" 2, roU "hit" 3, roF "bary" 4, roF "dm" 5, rwF "out" 6 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "j" (.member (v "tid") "x")
           , if_ (ge (v "j") (p "N")) [ ret ]
           , let_ uT "h" (at_ "hit" (v "j"))
           , if_ (eq (v "h") (u 4294967295))
               [ setAt "out" (v "j" * u 3) (fl 0.0)
               , setAt "out" (v "j" * u 3 + u 1) (fl 0.0)
               , setAt "out" (v "j" * u 3 + u 2) (fl 0.0)
               , ret ]
           , let_ uT "a" (at_ "tri" (v "h" * u 3) * u 3)
           , let_ uT "b" (at_ "tri" (v "h" * u 3 + u 1) * u 3)
           , let_ uT "c" (at_ "tri" (v "h" * u 3 + u 2) * u 3)
           , let_ fT "ba" (at_ "bary" (v "j" * u 3))
           , let_ fT "bb" (at_ "bary" (v "j" * u 3 + u 1))
           , let_ fT "bc" (at_ "bary" (v "j" * u 3 + u 2)) ] ++
           vec "d" (fun k => v "ba" * at_ "dm" (v "a" + u k) + v "bb" * at_ "dm" (v "b" + u k)
                              + v "bc" * at_ "dm" (v "c" + u k)) ++
           rot6 "R" "prm" (u 0) ++
           matVec "rd" "R" "d" ++
           [ let_ fT "sc" (at_ "prm" (u 9)) ] ++
           (r3.map fun k => setAt "out" (v "j" * u 3 + u k) (v "sc" * s "rd" k))) ] }

-- BEGIN PIN
def expected : String :=
"struct HfResampleParams {
  uint N;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfResampleParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> prm;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tri;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> hit;
[[vk::binding(4, 0)]]
StructuredBuffer<float> bary;
[[vk::binding(5, 0)]]
StructuredBuffer<float> dm;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> out;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.N)) {
    return;
  }
  uint h = hit[j];
  if ((h == 4294967295u)) {
    out[(j * 3u)] = 0.000000;
    out[((j * 3u) + 1u)] = 0.000000;
    out[((j * 3u) + 2u)] = 0.000000;
    return;
  }
  uint a = (tri[(h * 3u)] * 3u);
  uint b = (tri[((h * 3u) + 1u)] * 3u);
  uint c = (tri[((h * 3u) + 2u)] * 3u);
  float ba = bary[(j * 3u)];
  float bb = bary[((j * 3u) + 1u)];
  float bc = bary[((j * 3u) + 2u)];
  float d0 = (((ba * dm[(a + 0u)]) + (bb * dm[(b + 0u)])) + (bc * dm[(c + 0u)]));
  float d1 = (((ba * dm[(a + 1u)]) + (bb * dm[(b + 1u)])) + (bc * dm[(c + 1u)]));
  float d2 = (((ba * dm[(a + 2u)]) + (bb * dm[(b + 2u)])) + (bc * dm[(c + 2u)]));
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
  float rd0 = (((R0 * d0) + (R1 * d1)) + (R2 * d2));
  float rd1 = (((R3 * d0) + (R4 * d1)) + (R5 * d2));
  float rd2 = (((R6 * d0) + (R7 * d1)) + (R8 * d2));
  float sc = prm[9u];
  out[((j * 3u) + 0u)] = (sc * rd0);
  out[((j * 3u) + 1u)] = (sc * rd1);
  out[((j * 3u) + 2u)] = (sc * rd2);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.Resample
