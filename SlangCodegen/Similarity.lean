import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common

/-!
# `HeadFit.SlangCodegen.Similarity` — the rigid transform and scale

    pos[i] = s · R · v[i] + t

`R` is the 3×3 rotation of the 6D pair `prm[0 … 5]` (the first two rows,
re-orthonormalised; AGENTS.md rule 11), `t = prm[6 … 8]`, `s = prm[9]`.
Every thread rebuilds `R` from the six floats (straight-line code, no
local arrays), so the kernel needs no second pass.

Bindings (set 0):

  0  ConstantBuffer<HfSimParams> { uint P; }
  1  StructuredBuffer<float>   prm  (10)
  2  StructuredBuffer<float>   v    (P·3)
  3  RWStructuredBuffer<float> pos  (P·3)
-/

namespace HeadFit.SlangCodegen.Similarity

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

def shader : SlangShaderModule :=
  { structs := [ { name := "HfSimParams", fields := [fld "P" uT] } ]
  , globals := [ paramsCB "HfSimParams", roF "prm" 1, roF "v" 2, rwF "pos" 3 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "i" (.member (v "tid") "x")
           , if_ (ge (v "i") (p "P")) [ ret ] ] ++
           rot6 "R" "prm" (u 0) ++
           vec "x" (fun k => at_ "v" (v "i" * u 3 + u k)) ++
           matVec "rx" "R" "x" ++
           [ let_ fT "sc" (at_ "prm" (u 9)) ] ++
           (r3.map fun k => setAt "pos" (v "i" * u 3 + u k) (v "sc" * s "rx" k + at_ "prm" (u (6 + k))))) ] }

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
RWStructuredBuffer<float> pos;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
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
  float x0 = v[((i * 3u) + 0u)];
  float x1 = v[((i * 3u) + 1u)];
  float x2 = v[((i * 3u) + 2u)];
  float rx0 = (((R0 * x0) + (R1 * x1)) + (R2 * x2));
  float rx1 = (((R3 * x0) + (R4 * x1)) + (R5 * x2));
  float rx2 = (((R6 * x0) + (R7 * x1)) + (R8 * x2));
  float sc = prm[9u];
  pos[((i * 3u) + 0u)] = ((sc * rx0) + prm[6u]);
  pos[((i * 3u) + 1u)] = ((sc * rx1) + prm[7u]);
  pos[((i * 3u) + 2u)] = ((sc * rx2) + prm[8u]);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.Similarity
