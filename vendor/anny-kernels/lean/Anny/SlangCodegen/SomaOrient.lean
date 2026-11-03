import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common
import Anny.SlangCodegen.SomaCommon

/-!
# `Anny.SlangCodegen.SomaOrient` — joint orient, FK seeds and corrective features, one thread per public joint

SOMA-X `SOMALayer.pose` (procedural mode), per public joint j:

    R_j    = Oᵖᵀ_j · P_j · O_j                 apply_source_joint_orient
    acc_j  = [R_j | t_j],  t_j = transl if j = gtj else localT_j
    anc_j  = parent_j (0xFFFFFFFF for the root)  the seed of soma_jump
    X_j    = Bᵀ_j · R_j,  X00 −= 1, X11 −= 1     CorrectivesMLP.forward
    feat_j = X_j's first two columns, row by row (6 floats)

`P_j` is the input pose (T-pose relative, row 0 the identity SOMA-X pads
in), `O`/`Oᵖᵀ` are `source_joint_orient` and `source_joint_orient_parent_t`,
`localT` the public bind-local translations and `B` the correctives model's
bindpose. `R_j` is also written out: the twist channels read it.

Bindings (set 0):

   0  ConstantBuffer<SomaOrientParams> { uint NJ; uint gtj; }
   1  StructuredBuffer<float>   pose      (NJ·9)
   2  StructuredBuffer<float>   transl    (3)
   3  StructuredBuffer<float>   orient    (NJ·9)
   4  StructuredBuffer<float>   orientPT  (NJ·9)
   5  StructuredBuffer<float>   localT    (NJ·3)
   6  StructuredBuffer<uint>    parents   (NJ)
   7  StructuredBuffer<float>   corrBind  (NJ·9)
   8  RWStructuredBuffer<float> rabs      (NJ·9)
   9  RWStructuredBuffer<float> acc       (NJ·12)
  10  RWStructuredBuffer<uint>  anc       (NJ)
  11  RWStructuredBuffer<float> feat      (NJ·6)
-/

namespace Anny.SlangCodegen.SomaOrient

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common
open Anny.SlangCodegen.SomaCommon

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaOrientParams", fields := [fld "NJ" uT, fld "gtj" uT] } ]
  , globals :=
      [ paramsCB "SomaOrientParams", roF "pose" 1, roF "transl" 2, roF "orient" 3, roF "orientPT" 4
      , roF "localT" 5, roU "parents" 6, roF "corrBind" 7, rwF "rabs" 8, rwF "acc" 9
      , rwU "anc" 10, rwF "feat" 11 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "j" (.member (v "tid") "x")
           , if_ (ge (v "j") (p "NJ")) [ ret ]
           , let_ uT "o9" (v "j" * u 9)
           , let_ bT "g" (eq (v "j") (p "gtj")) ] ++
           load9 "P" "pose" (v "o9") ++ load9 "O" "orient" (v "o9") ++ load9 "Q" "orientPT" (v "o9") ++
           matMul "QP" "Q" "P" ++ matMul "R" "QP" "O" ++
           store9 "rabs" (v "o9") "R" ++
           vec "t" (fun k => sel (v "g") (at_ "transl" (u k)) (at_ "localT" (v "j" * u 3 + u k))) ++
           store12 "acc" (v "j" * u 12) "R" "t" ++
           [ setAt "anc" (v "j") (at_ "parents" (v "j")) ] ++
           load9 "B" "corrBind" (v "o9") ++ matTMul "X" "B" "R" ++
           [ let_ uT "o6" (v "j" * u 6)
           , setAt "feat" (v "o6") (s "X" 0 - fl 1.0)
           , setAt "feat" (v "o6" + u 1) (s "X" 1)
           , setAt "feat" (v "o6" + u 2) (s "X" 3)
           , setAt "feat" (v "o6" + u 3) (s "X" 4 - fl 1.0)
           , setAt "feat" (v "o6" + u 4) (s "X" 6)
           , setAt "feat" (v "o6" + u 5) (s "X" 7) ]) ] }

-- BEGIN PIN
def expected : String :=
"struct SomaOrientParams {
  uint NJ;
  uint gtj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaOrientParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> pose;
[[vk::binding(2, 0)]]
StructuredBuffer<float> transl;
[[vk::binding(3, 0)]]
StructuredBuffer<float> orient;
[[vk::binding(4, 0)]]
StructuredBuffer<float> orientPT;
[[vk::binding(5, 0)]]
StructuredBuffer<float> localT;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> parents;
[[vk::binding(7, 0)]]
StructuredBuffer<float> corrBind;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> rabs;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> acc;
[[vk::binding(10, 0)]]
RWStructuredBuffer<uint> anc;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float> feat;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.NJ)) {
    return;
  }
  uint o9 = (j * 9u);
  bool g = (j == params.gtj);
  float P0 = pose[(o9 + 0u)];
  float P1 = pose[(o9 + 1u)];
  float P2 = pose[(o9 + 2u)];
  float P3 = pose[(o9 + 3u)];
  float P4 = pose[(o9 + 4u)];
  float P5 = pose[(o9 + 5u)];
  float P6 = pose[(o9 + 6u)];
  float P7 = pose[(o9 + 7u)];
  float P8 = pose[(o9 + 8u)];
  float O0 = orient[(o9 + 0u)];
  float O1 = orient[(o9 + 1u)];
  float O2 = orient[(o9 + 2u)];
  float O3 = orient[(o9 + 3u)];
  float O4 = orient[(o9 + 4u)];
  float O5 = orient[(o9 + 5u)];
  float O6 = orient[(o9 + 6u)];
  float O7 = orient[(o9 + 7u)];
  float O8 = orient[(o9 + 8u)];
  float Q0 = orientPT[(o9 + 0u)];
  float Q1 = orientPT[(o9 + 1u)];
  float Q2 = orientPT[(o9 + 2u)];
  float Q3 = orientPT[(o9 + 3u)];
  float Q4 = orientPT[(o9 + 4u)];
  float Q5 = orientPT[(o9 + 5u)];
  float Q6 = orientPT[(o9 + 6u)];
  float Q7 = orientPT[(o9 + 7u)];
  float Q8 = orientPT[(o9 + 8u)];
  float QP0 = (((Q0 * P0) + (Q1 * P3)) + (Q2 * P6));
  float QP1 = (((Q0 * P1) + (Q1 * P4)) + (Q2 * P7));
  float QP2 = (((Q0 * P2) + (Q1 * P5)) + (Q2 * P8));
  float QP3 = (((Q3 * P0) + (Q4 * P3)) + (Q5 * P6));
  float QP4 = (((Q3 * P1) + (Q4 * P4)) + (Q5 * P7));
  float QP5 = (((Q3 * P2) + (Q4 * P5)) + (Q5 * P8));
  float QP6 = (((Q6 * P0) + (Q7 * P3)) + (Q8 * P6));
  float QP7 = (((Q6 * P1) + (Q7 * P4)) + (Q8 * P7));
  float QP8 = (((Q6 * P2) + (Q7 * P5)) + (Q8 * P8));
  float R0 = (((QP0 * O0) + (QP1 * O3)) + (QP2 * O6));
  float R1 = (((QP0 * O1) + (QP1 * O4)) + (QP2 * O7));
  float R2 = (((QP0 * O2) + (QP1 * O5)) + (QP2 * O8));
  float R3 = (((QP3 * O0) + (QP4 * O3)) + (QP5 * O6));
  float R4 = (((QP3 * O1) + (QP4 * O4)) + (QP5 * O7));
  float R5 = (((QP3 * O2) + (QP4 * O5)) + (QP5 * O8));
  float R6 = (((QP6 * O0) + (QP7 * O3)) + (QP8 * O6));
  float R7 = (((QP6 * O1) + (QP7 * O4)) + (QP8 * O7));
  float R8 = (((QP6 * O2) + (QP7 * O5)) + (QP8 * O8));
  rabs[(o9 + 0u)] = R0;
  rabs[(o9 + 1u)] = R1;
  rabs[(o9 + 2u)] = R2;
  rabs[(o9 + 3u)] = R3;
  rabs[(o9 + 4u)] = R4;
  rabs[(o9 + 5u)] = R5;
  rabs[(o9 + 6u)] = R6;
  rabs[(o9 + 7u)] = R7;
  rabs[(o9 + 8u)] = R8;
  float t0 = (g ? transl[0u] : localT[((j * 3u) + 0u)]);
  float t1 = (g ? transl[1u] : localT[((j * 3u) + 1u)]);
  float t2 = (g ? transl[2u] : localT[((j * 3u) + 2u)]);
  acc[((j * 12u) + 0u)] = R0;
  acc[((j * 12u) + 1u)] = R1;
  acc[((j * 12u) + 2u)] = R2;
  acc[((j * 12u) + 4u)] = R3;
  acc[((j * 12u) + 5u)] = R4;
  acc[((j * 12u) + 6u)] = R5;
  acc[((j * 12u) + 8u)] = R6;
  acc[((j * 12u) + 9u)] = R7;
  acc[((j * 12u) + 10u)] = R8;
  acc[((j * 12u) + 3u)] = t0;
  acc[((j * 12u) + 7u)] = t1;
  acc[((j * 12u) + 11u)] = t2;
  anc[j] = parents[j];
  float B0 = corrBind[(o9 + 0u)];
  float B1 = corrBind[(o9 + 1u)];
  float B2 = corrBind[(o9 + 2u)];
  float B3 = corrBind[(o9 + 3u)];
  float B4 = corrBind[(o9 + 4u)];
  float B5 = corrBind[(o9 + 5u)];
  float B6 = corrBind[(o9 + 6u)];
  float B7 = corrBind[(o9 + 7u)];
  float B8 = corrBind[(o9 + 8u)];
  float X0 = (((B0 * R0) + (B3 * R3)) + (B6 * R6));
  float X1 = (((B0 * R1) + (B3 * R4)) + (B6 * R7));
  float X2 = (((B0 * R2) + (B3 * R5)) + (B6 * R8));
  float X3 = (((B1 * R0) + (B4 * R3)) + (B7 * R6));
  float X4 = (((B1 * R1) + (B4 * R4)) + (B7 * R7));
  float X5 = (((B1 * R2) + (B4 * R5)) + (B7 * R8));
  float X6 = (((B2 * R0) + (B5 * R3)) + (B8 * R6));
  float X7 = (((B2 * R1) + (B5 * R4)) + (B8 * R7));
  float X8 = (((B2 * R2) + (B5 * R5)) + (B8 * R8));
  uint o6 = (j * 6u);
  feat[o6] = (X0 - 1.000000);
  feat[(o6 + 1u)] = X1;
  feat[(o6 + 2u)] = X3;
  feat[(o6 + 3u)] = (X4 - 1.000000);
  feat[(o6 + 4u)] = X6;
  feat[(o6 + 5u)] = X7;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaOrient
