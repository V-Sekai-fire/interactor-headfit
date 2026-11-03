import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common
import Anny.SlangCodegen.SomaCommon

/-!
# `Anny.SlangCodegen.SomaTwist` — the twist channels of every public joint, one thread per joint

`SOMAProceduralParameterTransform._twist_angles_from_source`'s three
extraction modes, each written to its own block of `chan`:

    chan[0·NJ + j]  local_x_euler          eulerChannel(R_j, axis_j) · sign_j
    chan[1·NJ + j]  local_x_swing_twist    twistAngle(R_j, axis_j)   · sign_j
    chan[2·NJ + j]  aligned_x_swing_twist  see below (0 for joints no segment names)

`R_j` is `soma_orient`'s `rabs` and `axis_j`/`sign_j` the source channel
SOMA-X assigns from the segments (axis 0, sign 1 elsewhere).

Aligned mode: for segment s and its joints k ∈ {end, start, parent},
`V_k = W_k · B_kᵀ · A_s`, with `W_k` the joint's FK world rotation, `B_k`
its T-pose world rotation and `A_s` the segment's bind alignment frame:
the matrix of SOMA-X's `q_current ⊗ q_bind⁻¹ ⊗ q_align` (a conjugate is
a transpose). The end joint gets the X twist of `V_startᵀ · V_end`; a
reverse segment's start joint gets the X twist of `V_parentᵀ · V_start`,
written after every end value as SOMA-X assigns them. No sign is applied,
as in SOMA-X.

`twistForm` picks the swing-twist angle. 0, the default, is the plain
swing-twist split from the IK solver's rank-1 Kabsch completion
(`SomaCommon.swingTwistAngle`, after swing-twist-kusudama `KusKabsch.lean`):
it returns the twist exactly at any swing. 1 is SOMA-X's half-angle form
(`SomaCommon.twistAngle`), which drifts from the twist as swing grows (3.8
degrees at 30 degrees of swing, 37 at 90); it is kept only so the parity
gate can check every other step against SOMA-X.

Bindings (set 0):

   0  ConstantBuffer<SomaTwistParams> { uint NJ; uint NS; uint twistForm; }
   1  StructuredBuffer<float>   rabs      (NJ·9)
   2  StructuredBuffer<float>   world     (NJ·12)  soma_jump's public world
   3  StructuredBuffer<uint>    srcAxis   (NJ)
   4  StructuredBuffer<float>   srcSign   (NJ)
   5  StructuredBuffer<uint>    segStart  (NS)
   6  StructuredBuffer<uint>    segEnd    (NS)
   7  StructuredBuffer<uint>    segParent (NS)
   8  StructuredBuffer<uint>    segRev    (NS)
   9  StructuredBuffer<float>   bindRot   (NJ·9)
  10  StructuredBuffer<float>   alignRot  (NS·9)
  11  RWStructuredBuffer<float> chan      (3·NJ)
-/

namespace Anny.SlangCodegen.SomaTwist

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common
open Anny.SlangCodegen.SomaCommon

/-- `pfx = W_k · B_kᵀ · A_s` for the joint index `k` and the segment `s`. -/
def virt (pfx : String) (k : E) : List St :=
  loadRot12 (pfx ++ "W") "world" (k * u 12) ++
  load9 (pfx ++ "B") "bindRot" (k * u 9) ++
  matMulT (pfx ++ "WB") (pfx ++ "W") (pfx ++ "B") ++
  matMul pfx (pfx ++ "WB") "A"

/-- `al = ` the X twist of `aᵀ · b`, for the joints `ka` and `kb` of segment `s`. -/
def relTwist (ka kb : E) : List St :=
  load9 "A" "alignRot" (v "s" * u 9) ++
  virt "Va" ka ++ virt "Vb" kb ++
  matTMul "Rel" "Va" "Vb" ++
  twistAngle "tw" "Rel" (u 0) ++
  swingTwistAngle "st" "Rel" (u 0) ++
  [ setv "al" (sel (eq (p "twistForm") (u 0)) (v "st") (v "tw")) ]

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaTwistParams", fields := [fld "NJ" uT, fld "NS" uT, fld "twistForm" uT] } ]
  , globals :=
      [ paramsCB "SomaTwistParams", roF "rabs" 1, roF "world" 2, roU "srcAxis" 3, roF "srcSign" 4
      , roU "segStart" 5, roU "segEnd" 6, roU "segParent" 7, roU "segRev" 8
      , roF "bindRot" 9, roF "alignRot" 10, rwF "chan" 11 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "j" (.member (v "tid") "x")
           , if_ (ge (v "j") (p "NJ")) [ ret ]
           , let_ uT "ax" (at_ "srcAxis" (v "j"))
           , let_ fT "sg" (at_ "srcSign" (v "j")) ] ++
           load9 "R" "rabs" (v "j" * u 9) ++
           [ setAt "chan" (v "j") (eulerChannel "R" (v "ax") * v "sg") ] ++
           twistAngle "sw" "R" (v "ax") ++
           swingTwistAngle "ss" "R" (v "ax") ++
           [ setAt "chan" (p "NJ" + v "j") (sel (eq (p "twistForm") (u 0)) (v "ss") (v "sw") * v "sg")
           , let_ fT "al" (fl 0.0)
           , for_ "s" (u 0) (p "NS")
               [ if_ (eq (at_ "segEnd" (v "s")) (v "j"))
                   (relTwist (at_ "segStart" (v "s")) (v "j")) ]
           , for_ "s" (u 0) (p "NS")
               [ if_ (and_ (ne (at_ "segRev" (v "s")) (u 0)) (eq (at_ "segStart" (v "s")) (v "j")))
                   (relTwist (at_ "segParent" (v "s")) (v "j")) ]
           , setAt "chan" (u 2 * p "NJ" + v "j") (v "al") ]) ] }

-- BEGIN PIN
def expected : String :=
"struct SomaTwistParams {
  uint NJ;
  uint NS;
  uint twistForm;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaTwistParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> rabs;
[[vk::binding(2, 0)]]
StructuredBuffer<float> world;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> srcAxis;
[[vk::binding(4, 0)]]
StructuredBuffer<float> srcSign;
[[vk::binding(5, 0)]]
StructuredBuffer<uint> segStart;
[[vk::binding(6, 0)]]
StructuredBuffer<uint> segEnd;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> segParent;
[[vk::binding(8, 0)]]
StructuredBuffer<uint> segRev;
[[vk::binding(9, 0)]]
StructuredBuffer<float> bindRot;
[[vk::binding(10, 0)]]
StructuredBuffer<float> alignRot;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float> chan;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.NJ)) {
    return;
  }
  uint ax = srcAxis[j];
  float sg = srcSign[j];
  float R0 = rabs[((j * 9u) + 0u)];
  float R1 = rabs[((j * 9u) + 1u)];
  float R2 = rabs[((j * 9u) + 2u)];
  float R3 = rabs[((j * 9u) + 3u)];
  float R4 = rabs[((j * 9u) + 4u)];
  float R5 = rabs[((j * 9u) + 5u)];
  float R6 = rabs[((j * 9u) + 6u)];
  float R7 = rabs[((j * 9u) + 7u)];
  float R8 = rabs[((j * 9u) + 8u)];
  chan[j] = (((ax == 0u) ? atan2(R7, R4) : ((ax == 1u) ? asin((-clamp(R6, -1.000000, 1.000000))) : atan2(R3, R0))) * sg);
  float sw_t = max((((1.000000 + R0) + R4) + R8), 0.000000);
  float sw = (4.000000 * atan2(((ax == 0u) ? (R7 - R5) : ((ax == 1u) ? (R2 - R6) : (R3 - R1))), (sw_t + (2.000000 * sqrt(sw_t)))));
  float ss_v00 = ((ax == 0u) ? 1.000000 : 0.000000);
  float ss_v01 = ((ax == 1u) ? 1.000000 : 0.000000);
  float ss_v02 = ((ax == 2u) ? 1.000000 : 0.000000);
  float ss_ru0 = (((R0 * ss_v00) + (R1 * ss_v01)) + (R2 * ss_v02));
  float ss_ru1 = (((R3 * ss_v00) + (R4 * ss_v01)) + (R5 * ss_v02));
  float ss_ru2 = (((R6 * ss_v00) + (R7 * ss_v01)) + (R8 * ss_v02));
  float ss_u0_l = sqrt((((ss_ru0 * ss_ru0) + (ss_ru1 * ss_ru1)) + (ss_ru2 * ss_ru2)));
  float ss_u00 = ((ss_u0_l > 1.0e-12f) ? (ss_ru0 / ss_u0_l) : ss_ru0);
  float ss_u01 = ((ss_u0_l > 1.0e-12f) ? (ss_ru1 / ss_u0_l) : ss_ru1);
  float ss_u02 = ((ss_u0_l > 1.0e-12f) ? (ss_ru2 / ss_u0_l) : ss_ru2);
  bool ss_fy = (abs(ss_v01) < 0.900000);
  float ss_ref0 = (ss_fy ? 0.000000 : 1.000000);
  float ss_ref1 = (ss_fy ? 1.000000 : 0.000000);
  float ss_ref2 = 0.000000;
  float ss_v1r0 = ((ss_v01 * ss_ref2) - (ss_v02 * ss_ref1));
  float ss_v1r1 = ((ss_v02 * ss_ref0) - (ss_v00 * ss_ref2));
  float ss_v1r2 = ((ss_v00 * ss_ref1) - (ss_v01 * ss_ref0));
  float ss_v1_l = sqrt((((ss_v1r0 * ss_v1r0) + (ss_v1r1 * ss_v1r1)) + (ss_v1r2 * ss_v1r2)));
  float ss_v10 = ((ss_v1_l > 1.0e-12f) ? (ss_v1r0 / ss_v1_l) : ss_v1r0);
  float ss_v11 = ((ss_v1_l > 1.0e-12f) ? (ss_v1r1 / ss_v1_l) : ss_v1r1);
  float ss_v12 = ((ss_v1_l > 1.0e-12f) ? (ss_v1r2 / ss_v1_l) : ss_v1r2);
  float ss_v2r0 = ((ss_v01 * ss_v12) - (ss_v02 * ss_v11));
  float ss_v2r1 = ((ss_v02 * ss_v10) - (ss_v00 * ss_v12));
  float ss_v2r2 = ((ss_v00 * ss_v11) - (ss_v01 * ss_v10));
  float ss_v2_l = sqrt((((ss_v2r0 * ss_v2r0) + (ss_v2r1 * ss_v2r1)) + (ss_v2r2 * ss_v2r2)));
  float ss_v20 = ((ss_v2_l > 1.0e-12f) ? (ss_v2r0 / ss_v2_l) : ss_v2r0);
  float ss_v21 = ((ss_v2_l > 1.0e-12f) ? (ss_v2r1 / ss_v2_l) : ss_v2r1);
  float ss_v22 = ((ss_v2_l > 1.0e-12f) ? (ss_v2r2 / ss_v2_l) : ss_v2r2);
  float ss_ax0 = ((ss_v01 * ss_u02) - (ss_v02 * ss_u01));
  float ss_ax1 = ((ss_v02 * ss_u00) - (ss_v00 * ss_u02));
  float ss_ax2 = ((ss_v00 * ss_u01) - (ss_v01 * ss_u00));
  float ss_sn = sqrt((((ss_ax0 * ss_ax0) + (ss_ax1 * ss_ax1)) + (ss_ax2 * ss_ax2)));
  float ss_cs = (((ss_v00 * ss_u00) + (ss_v01 * ss_u01)) + (ss_v02 * ss_u02));
  float ss_k0 = (ss_ax0 / max(ss_sn, 1.0e-9f));
  float ss_k1 = (ss_ax1 / max(ss_sn, 1.0e-9f));
  float ss_k2 = (ss_ax2 / max(ss_sn, 1.0e-9f));
  float ss_c1_kx0 = ((ss_k1 * ss_v12) - (ss_k2 * ss_v11));
  float ss_c1_kx1 = ((ss_k2 * ss_v10) - (ss_k0 * ss_v12));
  float ss_c1_kx2 = ((ss_k0 * ss_v11) - (ss_k1 * ss_v10));
  float ss_c1_kw = (((ss_k0 * ss_v10) + (ss_k1 * ss_v11)) + (ss_k2 * ss_v12));
  float ss_c10 = (((ss_cs * ss_v10) + (ss_sn * ss_c1_kx0)) + ((ss_c1_kw * (1.000000 - ss_cs)) * ss_k0));
  float ss_c11 = (((ss_cs * ss_v11) + (ss_sn * ss_c1_kx1)) + ((ss_c1_kw * (1.000000 - ss_cs)) * ss_k1));
  float ss_c12 = (((ss_cs * ss_v12) + (ss_sn * ss_c1_kx2)) + ((ss_c1_kw * (1.000000 - ss_cs)) * ss_k2));
  float ss_c2_kx0 = ((ss_k1 * ss_v22) - (ss_k2 * ss_v21));
  float ss_c2_kx1 = ((ss_k2 * ss_v20) - (ss_k0 * ss_v22));
  float ss_c2_kx2 = ((ss_k0 * ss_v21) - (ss_k1 * ss_v20));
  float ss_c2_kw = (((ss_k0 * ss_v20) + (ss_k1 * ss_v21)) + (ss_k2 * ss_v22));
  float ss_c20 = (((ss_cs * ss_v20) + (ss_sn * ss_c2_kx0)) + ((ss_c2_kw * (1.000000 - ss_cs)) * ss_k0));
  float ss_c21 = (((ss_cs * ss_v21) + (ss_sn * ss_c2_kx1)) + ((ss_c2_kw * (1.000000 - ss_cs)) * ss_k1));
  float ss_c22 = (((ss_cs * ss_v22) + (ss_sn * ss_c2_kx2)) + ((ss_c2_kw * (1.000000 - ss_cs)) * ss_k2));
  if ((ss_sn < 1.0e-9f)) {
    if ((ss_cs > 0.000000)) {
      ss_c10 = ss_v10;
      ss_c11 = ss_v11;
      ss_c12 = ss_v12;
      ss_c20 = ss_v20;
      ss_c21 = ss_v21;
      ss_c22 = ss_v22;
    } else {
      bool ss_px = ((abs(ss_u00) <= abs(ss_u01)) && (abs(ss_u00) <= abs(ss_u02)));
      float ss_pr0 = (ss_px ? 1.000000 : 0.000000);
      float ss_pr1 = (ss_px ? 0.000000 : 1.000000);
      float ss_pr2 = 0.000000;
      float ss_p10 = ((ss_u01 * ss_pr2) - (ss_u02 * ss_pr1));
      float ss_p11 = ((ss_u02 * ss_pr0) - (ss_u00 * ss_pr2));
      float ss_p12 = ((ss_u00 * ss_pr1) - (ss_u01 * ss_pr0));
      float ss_p1n_l = sqrt((((ss_p10 * ss_p10) + (ss_p11 * ss_p11)) + (ss_p12 * ss_p12)));
      float ss_p1n0 = ((ss_p1n_l > 1.0e-12f) ? (ss_p10 / ss_p1n_l) : ss_p10);
      float ss_p1n1 = ((ss_p1n_l > 1.0e-12f) ? (ss_p11 / ss_p1n_l) : ss_p11);
      float ss_p1n2 = ((ss_p1n_l > 1.0e-12f) ? (ss_p12 / ss_p1n_l) : ss_p12);
      float ss_p20 = ((ss_u01 * ss_p1n2) - (ss_u02 * ss_p1n1));
      float ss_p21 = ((ss_u02 * ss_p1n0) - (ss_u00 * ss_p1n2));
      float ss_p22 = ((ss_u00 * ss_p1n1) - (ss_u01 * ss_p1n0));
      float ss_p2n_l = sqrt((((ss_p20 * ss_p20) + (ss_p21 * ss_p21)) + (ss_p22 * ss_p22)));
      float ss_p2n0 = ((ss_p2n_l > 1.0e-12f) ? (ss_p20 / ss_p2n_l) : ss_p20);
      float ss_p2n1 = ((ss_p2n_l > 1.0e-12f) ? (ss_p21 / ss_p2n_l) : ss_p21);
      float ss_p2n2 = ((ss_p2n_l > 1.0e-12f) ? (ss_p22 / ss_p2n_l) : ss_p22);
      ss_c10 = ss_p1n0;
      ss_c11 = ss_p1n1;
      ss_c12 = ss_p1n2;
      ss_c20 = ss_p2n0;
      ss_c21 = ss_p2n1;
      ss_c22 = ss_p2n2;
    }
  }
  float ss_R00 = (((ss_u00 * ss_v00) + (ss_c10 * ss_v10)) + (ss_c20 * ss_v20));
  float ss_R01 = (((ss_u00 * ss_v01) + (ss_c10 * ss_v11)) + (ss_c20 * ss_v21));
  float ss_R02 = (((ss_u00 * ss_v02) + (ss_c10 * ss_v12)) + (ss_c20 * ss_v22));
  float ss_R03 = (((ss_u01 * ss_v00) + (ss_c11 * ss_v10)) + (ss_c21 * ss_v20));
  float ss_R04 = (((ss_u01 * ss_v01) + (ss_c11 * ss_v11)) + (ss_c21 * ss_v21));
  float ss_R05 = (((ss_u01 * ss_v02) + (ss_c11 * ss_v12)) + (ss_c21 * ss_v22));
  float ss_R06 = (((ss_u02 * ss_v00) + (ss_c12 * ss_v10)) + (ss_c22 * ss_v20));
  float ss_R07 = (((ss_u02 * ss_v01) + (ss_c12 * ss_v11)) + (ss_c22 * ss_v21));
  float ss_R08 = (((ss_u02 * ss_v02) + (ss_c12 * ss_v12)) + (ss_c22 * ss_v22));
  float ss_T0 = (((ss_R00 * R0) + (ss_R03 * R3)) + (ss_R06 * R6));
  float ss_T1 = (((ss_R00 * R1) + (ss_R03 * R4)) + (ss_R06 * R7));
  float ss_T2 = (((ss_R00 * R2) + (ss_R03 * R5)) + (ss_R06 * R8));
  float ss_T3 = (((ss_R01 * R0) + (ss_R04 * R3)) + (ss_R07 * R6));
  float ss_T4 = (((ss_R01 * R1) + (ss_R04 * R4)) + (ss_R07 * R7));
  float ss_T5 = (((ss_R01 * R2) + (ss_R04 * R5)) + (ss_R07 * R8));
  float ss_T6 = (((ss_R02 * R0) + (ss_R05 * R3)) + (ss_R08 * R6));
  float ss_T7 = (((ss_R02 * R1) + (ss_R05 * R4)) + (ss_R08 * R7));
  float ss_T8 = (((ss_R02 * R2) + (ss_R05 * R5)) + (ss_R08 * R8));
  float ss = atan2(((ax == 0u) ? (ss_T7 - ss_T5) : ((ax == 1u) ? (ss_T2 - ss_T6) : (ss_T3 - ss_T1))), ((ax == 0u) ? (ss_T4 + ss_T8) : ((ax == 1u) ? (ss_T8 + ss_T0) : (ss_T0 + ss_T4))));
  chan[(params.NJ + j)] = (((params.twistForm == 0u) ? ss : sw) * sg);
  float al = 0.000000;
  for (uint s = 0u; s < params.NS; ++s) {
    if ((segEnd[s] == j)) {
      float A0 = alignRot[((s * 9u) + 0u)];
      float A1 = alignRot[((s * 9u) + 1u)];
      float A2 = alignRot[((s * 9u) + 2u)];
      float A3 = alignRot[((s * 9u) + 3u)];
      float A4 = alignRot[((s * 9u) + 4u)];
      float A5 = alignRot[((s * 9u) + 5u)];
      float A6 = alignRot[((s * 9u) + 6u)];
      float A7 = alignRot[((s * 9u) + 7u)];
      float A8 = alignRot[((s * 9u) + 8u)];
      float VaW0 = world[((segStart[s] * 12u) + 0u)];
      float VaW1 = world[((segStart[s] * 12u) + 1u)];
      float VaW2 = world[((segStart[s] * 12u) + 2u)];
      float VaW3 = world[((segStart[s] * 12u) + 4u)];
      float VaW4 = world[((segStart[s] * 12u) + 5u)];
      float VaW5 = world[((segStart[s] * 12u) + 6u)];
      float VaW6 = world[((segStart[s] * 12u) + 8u)];
      float VaW7 = world[((segStart[s] * 12u) + 9u)];
      float VaW8 = world[((segStart[s] * 12u) + 10u)];
      float VaB0 = bindRot[((segStart[s] * 9u) + 0u)];
      float VaB1 = bindRot[((segStart[s] * 9u) + 1u)];
      float VaB2 = bindRot[((segStart[s] * 9u) + 2u)];
      float VaB3 = bindRot[((segStart[s] * 9u) + 3u)];
      float VaB4 = bindRot[((segStart[s] * 9u) + 4u)];
      float VaB5 = bindRot[((segStart[s] * 9u) + 5u)];
      float VaB6 = bindRot[((segStart[s] * 9u) + 6u)];
      float VaB7 = bindRot[((segStart[s] * 9u) + 7u)];
      float VaB8 = bindRot[((segStart[s] * 9u) + 8u)];
      float VaWB0 = (((VaW0 * VaB0) + (VaW1 * VaB1)) + (VaW2 * VaB2));
      float VaWB1 = (((VaW0 * VaB3) + (VaW1 * VaB4)) + (VaW2 * VaB5));
      float VaWB2 = (((VaW0 * VaB6) + (VaW1 * VaB7)) + (VaW2 * VaB8));
      float VaWB3 = (((VaW3 * VaB0) + (VaW4 * VaB1)) + (VaW5 * VaB2));
      float VaWB4 = (((VaW3 * VaB3) + (VaW4 * VaB4)) + (VaW5 * VaB5));
      float VaWB5 = (((VaW3 * VaB6) + (VaW4 * VaB7)) + (VaW5 * VaB8));
      float VaWB6 = (((VaW6 * VaB0) + (VaW7 * VaB1)) + (VaW8 * VaB2));
      float VaWB7 = (((VaW6 * VaB3) + (VaW7 * VaB4)) + (VaW8 * VaB5));
      float VaWB8 = (((VaW6 * VaB6) + (VaW7 * VaB7)) + (VaW8 * VaB8));
      float Va0 = (((VaWB0 * A0) + (VaWB1 * A3)) + (VaWB2 * A6));
      float Va1 = (((VaWB0 * A1) + (VaWB1 * A4)) + (VaWB2 * A7));
      float Va2 = (((VaWB0 * A2) + (VaWB1 * A5)) + (VaWB2 * A8));
      float Va3 = (((VaWB3 * A0) + (VaWB4 * A3)) + (VaWB5 * A6));
      float Va4 = (((VaWB3 * A1) + (VaWB4 * A4)) + (VaWB5 * A7));
      float Va5 = (((VaWB3 * A2) + (VaWB4 * A5)) + (VaWB5 * A8));
      float Va6 = (((VaWB6 * A0) + (VaWB7 * A3)) + (VaWB8 * A6));
      float Va7 = (((VaWB6 * A1) + (VaWB7 * A4)) + (VaWB8 * A7));
      float Va8 = (((VaWB6 * A2) + (VaWB7 * A5)) + (VaWB8 * A8));
      float VbW0 = world[((j * 12u) + 0u)];
      float VbW1 = world[((j * 12u) + 1u)];
      float VbW2 = world[((j * 12u) + 2u)];
      float VbW3 = world[((j * 12u) + 4u)];
      float VbW4 = world[((j * 12u) + 5u)];
      float VbW5 = world[((j * 12u) + 6u)];
      float VbW6 = world[((j * 12u) + 8u)];
      float VbW7 = world[((j * 12u) + 9u)];
      float VbW8 = world[((j * 12u) + 10u)];
      float VbB0 = bindRot[((j * 9u) + 0u)];
      float VbB1 = bindRot[((j * 9u) + 1u)];
      float VbB2 = bindRot[((j * 9u) + 2u)];
      float VbB3 = bindRot[((j * 9u) + 3u)];
      float VbB4 = bindRot[((j * 9u) + 4u)];
      float VbB5 = bindRot[((j * 9u) + 5u)];
      float VbB6 = bindRot[((j * 9u) + 6u)];
      float VbB7 = bindRot[((j * 9u) + 7u)];
      float VbB8 = bindRot[((j * 9u) + 8u)];
      float VbWB0 = (((VbW0 * VbB0) + (VbW1 * VbB1)) + (VbW2 * VbB2));
      float VbWB1 = (((VbW0 * VbB3) + (VbW1 * VbB4)) + (VbW2 * VbB5));
      float VbWB2 = (((VbW0 * VbB6) + (VbW1 * VbB7)) + (VbW2 * VbB8));
      float VbWB3 = (((VbW3 * VbB0) + (VbW4 * VbB1)) + (VbW5 * VbB2));
      float VbWB4 = (((VbW3 * VbB3) + (VbW4 * VbB4)) + (VbW5 * VbB5));
      float VbWB5 = (((VbW3 * VbB6) + (VbW4 * VbB7)) + (VbW5 * VbB8));
      float VbWB6 = (((VbW6 * VbB0) + (VbW7 * VbB1)) + (VbW8 * VbB2));
      float VbWB7 = (((VbW6 * VbB3) + (VbW7 * VbB4)) + (VbW8 * VbB5));
      float VbWB8 = (((VbW6 * VbB6) + (VbW7 * VbB7)) + (VbW8 * VbB8));
      float Vb0 = (((VbWB0 * A0) + (VbWB1 * A3)) + (VbWB2 * A6));
      float Vb1 = (((VbWB0 * A1) + (VbWB1 * A4)) + (VbWB2 * A7));
      float Vb2 = (((VbWB0 * A2) + (VbWB1 * A5)) + (VbWB2 * A8));
      float Vb3 = (((VbWB3 * A0) + (VbWB4 * A3)) + (VbWB5 * A6));
      float Vb4 = (((VbWB3 * A1) + (VbWB4 * A4)) + (VbWB5 * A7));
      float Vb5 = (((VbWB3 * A2) + (VbWB4 * A5)) + (VbWB5 * A8));
      float Vb6 = (((VbWB6 * A0) + (VbWB7 * A3)) + (VbWB8 * A6));
      float Vb7 = (((VbWB6 * A1) + (VbWB7 * A4)) + (VbWB8 * A7));
      float Vb8 = (((VbWB6 * A2) + (VbWB7 * A5)) + (VbWB8 * A8));
      float Rel0 = (((Va0 * Vb0) + (Va3 * Vb3)) + (Va6 * Vb6));
      float Rel1 = (((Va0 * Vb1) + (Va3 * Vb4)) + (Va6 * Vb7));
      float Rel2 = (((Va0 * Vb2) + (Va3 * Vb5)) + (Va6 * Vb8));
      float Rel3 = (((Va1 * Vb0) + (Va4 * Vb3)) + (Va7 * Vb6));
      float Rel4 = (((Va1 * Vb1) + (Va4 * Vb4)) + (Va7 * Vb7));
      float Rel5 = (((Va1 * Vb2) + (Va4 * Vb5)) + (Va7 * Vb8));
      float Rel6 = (((Va2 * Vb0) + (Va5 * Vb3)) + (Va8 * Vb6));
      float Rel7 = (((Va2 * Vb1) + (Va5 * Vb4)) + (Va8 * Vb7));
      float Rel8 = (((Va2 * Vb2) + (Va5 * Vb5)) + (Va8 * Vb8));
      float tw_t = max((((1.000000 + Rel0) + Rel4) + Rel8), 0.000000);
      float tw = (4.000000 * atan2(((0u == 0u) ? (Rel7 - Rel5) : ((0u == 1u) ? (Rel2 - Rel6) : (Rel3 - Rel1))), (tw_t + (2.000000 * sqrt(tw_t)))));
      float st_v00 = ((0u == 0u) ? 1.000000 : 0.000000);
      float st_v01 = ((0u == 1u) ? 1.000000 : 0.000000);
      float st_v02 = ((0u == 2u) ? 1.000000 : 0.000000);
      float st_ru0 = (((Rel0 * st_v00) + (Rel1 * st_v01)) + (Rel2 * st_v02));
      float st_ru1 = (((Rel3 * st_v00) + (Rel4 * st_v01)) + (Rel5 * st_v02));
      float st_ru2 = (((Rel6 * st_v00) + (Rel7 * st_v01)) + (Rel8 * st_v02));
      float st_u0_l = sqrt((((st_ru0 * st_ru0) + (st_ru1 * st_ru1)) + (st_ru2 * st_ru2)));
      float st_u00 = ((st_u0_l > 1.0e-12f) ? (st_ru0 / st_u0_l) : st_ru0);
      float st_u01 = ((st_u0_l > 1.0e-12f) ? (st_ru1 / st_u0_l) : st_ru1);
      float st_u02 = ((st_u0_l > 1.0e-12f) ? (st_ru2 / st_u0_l) : st_ru2);
      bool st_fy = (abs(st_v01) < 0.900000);
      float st_ref0 = (st_fy ? 0.000000 : 1.000000);
      float st_ref1 = (st_fy ? 1.000000 : 0.000000);
      float st_ref2 = 0.000000;
      float st_v1r0 = ((st_v01 * st_ref2) - (st_v02 * st_ref1));
      float st_v1r1 = ((st_v02 * st_ref0) - (st_v00 * st_ref2));
      float st_v1r2 = ((st_v00 * st_ref1) - (st_v01 * st_ref0));
      float st_v1_l = sqrt((((st_v1r0 * st_v1r0) + (st_v1r1 * st_v1r1)) + (st_v1r2 * st_v1r2)));
      float st_v10 = ((st_v1_l > 1.0e-12f) ? (st_v1r0 / st_v1_l) : st_v1r0);
      float st_v11 = ((st_v1_l > 1.0e-12f) ? (st_v1r1 / st_v1_l) : st_v1r1);
      float st_v12 = ((st_v1_l > 1.0e-12f) ? (st_v1r2 / st_v1_l) : st_v1r2);
      float st_v2r0 = ((st_v01 * st_v12) - (st_v02 * st_v11));
      float st_v2r1 = ((st_v02 * st_v10) - (st_v00 * st_v12));
      float st_v2r2 = ((st_v00 * st_v11) - (st_v01 * st_v10));
      float st_v2_l = sqrt((((st_v2r0 * st_v2r0) + (st_v2r1 * st_v2r1)) + (st_v2r2 * st_v2r2)));
      float st_v20 = ((st_v2_l > 1.0e-12f) ? (st_v2r0 / st_v2_l) : st_v2r0);
      float st_v21 = ((st_v2_l > 1.0e-12f) ? (st_v2r1 / st_v2_l) : st_v2r1);
      float st_v22 = ((st_v2_l > 1.0e-12f) ? (st_v2r2 / st_v2_l) : st_v2r2);
      float st_ax0 = ((st_v01 * st_u02) - (st_v02 * st_u01));
      float st_ax1 = ((st_v02 * st_u00) - (st_v00 * st_u02));
      float st_ax2 = ((st_v00 * st_u01) - (st_v01 * st_u00));
      float st_sn = sqrt((((st_ax0 * st_ax0) + (st_ax1 * st_ax1)) + (st_ax2 * st_ax2)));
      float st_cs = (((st_v00 * st_u00) + (st_v01 * st_u01)) + (st_v02 * st_u02));
      float st_k0 = (st_ax0 / max(st_sn, 1.0e-9f));
      float st_k1 = (st_ax1 / max(st_sn, 1.0e-9f));
      float st_k2 = (st_ax2 / max(st_sn, 1.0e-9f));
      float st_c1_kx0 = ((st_k1 * st_v12) - (st_k2 * st_v11));
      float st_c1_kx1 = ((st_k2 * st_v10) - (st_k0 * st_v12));
      float st_c1_kx2 = ((st_k0 * st_v11) - (st_k1 * st_v10));
      float st_c1_kw = (((st_k0 * st_v10) + (st_k1 * st_v11)) + (st_k2 * st_v12));
      float st_c10 = (((st_cs * st_v10) + (st_sn * st_c1_kx0)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k0));
      float st_c11 = (((st_cs * st_v11) + (st_sn * st_c1_kx1)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k1));
      float st_c12 = (((st_cs * st_v12) + (st_sn * st_c1_kx2)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k2));
      float st_c2_kx0 = ((st_k1 * st_v22) - (st_k2 * st_v21));
      float st_c2_kx1 = ((st_k2 * st_v20) - (st_k0 * st_v22));
      float st_c2_kx2 = ((st_k0 * st_v21) - (st_k1 * st_v20));
      float st_c2_kw = (((st_k0 * st_v20) + (st_k1 * st_v21)) + (st_k2 * st_v22));
      float st_c20 = (((st_cs * st_v20) + (st_sn * st_c2_kx0)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k0));
      float st_c21 = (((st_cs * st_v21) + (st_sn * st_c2_kx1)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k1));
      float st_c22 = (((st_cs * st_v22) + (st_sn * st_c2_kx2)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k2));
      if ((st_sn < 1.0e-9f)) {
        if ((st_cs > 0.000000)) {
          st_c10 = st_v10;
          st_c11 = st_v11;
          st_c12 = st_v12;
          st_c20 = st_v20;
          st_c21 = st_v21;
          st_c22 = st_v22;
        } else {
          bool st_px = ((abs(st_u00) <= abs(st_u01)) && (abs(st_u00) <= abs(st_u02)));
          float st_pr0 = (st_px ? 1.000000 : 0.000000);
          float st_pr1 = (st_px ? 0.000000 : 1.000000);
          float st_pr2 = 0.000000;
          float st_p10 = ((st_u01 * st_pr2) - (st_u02 * st_pr1));
          float st_p11 = ((st_u02 * st_pr0) - (st_u00 * st_pr2));
          float st_p12 = ((st_u00 * st_pr1) - (st_u01 * st_pr0));
          float st_p1n_l = sqrt((((st_p10 * st_p10) + (st_p11 * st_p11)) + (st_p12 * st_p12)));
          float st_p1n0 = ((st_p1n_l > 1.0e-12f) ? (st_p10 / st_p1n_l) : st_p10);
          float st_p1n1 = ((st_p1n_l > 1.0e-12f) ? (st_p11 / st_p1n_l) : st_p11);
          float st_p1n2 = ((st_p1n_l > 1.0e-12f) ? (st_p12 / st_p1n_l) : st_p12);
          float st_p20 = ((st_u01 * st_p1n2) - (st_u02 * st_p1n1));
          float st_p21 = ((st_u02 * st_p1n0) - (st_u00 * st_p1n2));
          float st_p22 = ((st_u00 * st_p1n1) - (st_u01 * st_p1n0));
          float st_p2n_l = sqrt((((st_p20 * st_p20) + (st_p21 * st_p21)) + (st_p22 * st_p22)));
          float st_p2n0 = ((st_p2n_l > 1.0e-12f) ? (st_p20 / st_p2n_l) : st_p20);
          float st_p2n1 = ((st_p2n_l > 1.0e-12f) ? (st_p21 / st_p2n_l) : st_p21);
          float st_p2n2 = ((st_p2n_l > 1.0e-12f) ? (st_p22 / st_p2n_l) : st_p22);
          st_c10 = st_p1n0;
          st_c11 = st_p1n1;
          st_c12 = st_p1n2;
          st_c20 = st_p2n0;
          st_c21 = st_p2n1;
          st_c22 = st_p2n2;
        }
      }
      float st_R00 = (((st_u00 * st_v00) + (st_c10 * st_v10)) + (st_c20 * st_v20));
      float st_R01 = (((st_u00 * st_v01) + (st_c10 * st_v11)) + (st_c20 * st_v21));
      float st_R02 = (((st_u00 * st_v02) + (st_c10 * st_v12)) + (st_c20 * st_v22));
      float st_R03 = (((st_u01 * st_v00) + (st_c11 * st_v10)) + (st_c21 * st_v20));
      float st_R04 = (((st_u01 * st_v01) + (st_c11 * st_v11)) + (st_c21 * st_v21));
      float st_R05 = (((st_u01 * st_v02) + (st_c11 * st_v12)) + (st_c21 * st_v22));
      float st_R06 = (((st_u02 * st_v00) + (st_c12 * st_v10)) + (st_c22 * st_v20));
      float st_R07 = (((st_u02 * st_v01) + (st_c12 * st_v11)) + (st_c22 * st_v21));
      float st_R08 = (((st_u02 * st_v02) + (st_c12 * st_v12)) + (st_c22 * st_v22));
      float st_T0 = (((st_R00 * Rel0) + (st_R03 * Rel3)) + (st_R06 * Rel6));
      float st_T1 = (((st_R00 * Rel1) + (st_R03 * Rel4)) + (st_R06 * Rel7));
      float st_T2 = (((st_R00 * Rel2) + (st_R03 * Rel5)) + (st_R06 * Rel8));
      float st_T3 = (((st_R01 * Rel0) + (st_R04 * Rel3)) + (st_R07 * Rel6));
      float st_T4 = (((st_R01 * Rel1) + (st_R04 * Rel4)) + (st_R07 * Rel7));
      float st_T5 = (((st_R01 * Rel2) + (st_R04 * Rel5)) + (st_R07 * Rel8));
      float st_T6 = (((st_R02 * Rel0) + (st_R05 * Rel3)) + (st_R08 * Rel6));
      float st_T7 = (((st_R02 * Rel1) + (st_R05 * Rel4)) + (st_R08 * Rel7));
      float st_T8 = (((st_R02 * Rel2) + (st_R05 * Rel5)) + (st_R08 * Rel8));
      float st = atan2(((0u == 0u) ? (st_T7 - st_T5) : ((0u == 1u) ? (st_T2 - st_T6) : (st_T3 - st_T1))), ((0u == 0u) ? (st_T4 + st_T8) : ((0u == 1u) ? (st_T8 + st_T0) : (st_T0 + st_T4))));
      al = ((params.twistForm == 0u) ? st : tw);
    }
  }
  for (uint s = 0u; s < params.NS; ++s) {
    if (((segRev[s] != 0u) && (segStart[s] == j))) {
      float A0 = alignRot[((s * 9u) + 0u)];
      float A1 = alignRot[((s * 9u) + 1u)];
      float A2 = alignRot[((s * 9u) + 2u)];
      float A3 = alignRot[((s * 9u) + 3u)];
      float A4 = alignRot[((s * 9u) + 4u)];
      float A5 = alignRot[((s * 9u) + 5u)];
      float A6 = alignRot[((s * 9u) + 6u)];
      float A7 = alignRot[((s * 9u) + 7u)];
      float A8 = alignRot[((s * 9u) + 8u)];
      float VaW0 = world[((segParent[s] * 12u) + 0u)];
      float VaW1 = world[((segParent[s] * 12u) + 1u)];
      float VaW2 = world[((segParent[s] * 12u) + 2u)];
      float VaW3 = world[((segParent[s] * 12u) + 4u)];
      float VaW4 = world[((segParent[s] * 12u) + 5u)];
      float VaW5 = world[((segParent[s] * 12u) + 6u)];
      float VaW6 = world[((segParent[s] * 12u) + 8u)];
      float VaW7 = world[((segParent[s] * 12u) + 9u)];
      float VaW8 = world[((segParent[s] * 12u) + 10u)];
      float VaB0 = bindRot[((segParent[s] * 9u) + 0u)];
      float VaB1 = bindRot[((segParent[s] * 9u) + 1u)];
      float VaB2 = bindRot[((segParent[s] * 9u) + 2u)];
      float VaB3 = bindRot[((segParent[s] * 9u) + 3u)];
      float VaB4 = bindRot[((segParent[s] * 9u) + 4u)];
      float VaB5 = bindRot[((segParent[s] * 9u) + 5u)];
      float VaB6 = bindRot[((segParent[s] * 9u) + 6u)];
      float VaB7 = bindRot[((segParent[s] * 9u) + 7u)];
      float VaB8 = bindRot[((segParent[s] * 9u) + 8u)];
      float VaWB0 = (((VaW0 * VaB0) + (VaW1 * VaB1)) + (VaW2 * VaB2));
      float VaWB1 = (((VaW0 * VaB3) + (VaW1 * VaB4)) + (VaW2 * VaB5));
      float VaWB2 = (((VaW0 * VaB6) + (VaW1 * VaB7)) + (VaW2 * VaB8));
      float VaWB3 = (((VaW3 * VaB0) + (VaW4 * VaB1)) + (VaW5 * VaB2));
      float VaWB4 = (((VaW3 * VaB3) + (VaW4 * VaB4)) + (VaW5 * VaB5));
      float VaWB5 = (((VaW3 * VaB6) + (VaW4 * VaB7)) + (VaW5 * VaB8));
      float VaWB6 = (((VaW6 * VaB0) + (VaW7 * VaB1)) + (VaW8 * VaB2));
      float VaWB7 = (((VaW6 * VaB3) + (VaW7 * VaB4)) + (VaW8 * VaB5));
      float VaWB8 = (((VaW6 * VaB6) + (VaW7 * VaB7)) + (VaW8 * VaB8));
      float Va0 = (((VaWB0 * A0) + (VaWB1 * A3)) + (VaWB2 * A6));
      float Va1 = (((VaWB0 * A1) + (VaWB1 * A4)) + (VaWB2 * A7));
      float Va2 = (((VaWB0 * A2) + (VaWB1 * A5)) + (VaWB2 * A8));
      float Va3 = (((VaWB3 * A0) + (VaWB4 * A3)) + (VaWB5 * A6));
      float Va4 = (((VaWB3 * A1) + (VaWB4 * A4)) + (VaWB5 * A7));
      float Va5 = (((VaWB3 * A2) + (VaWB4 * A5)) + (VaWB5 * A8));
      float Va6 = (((VaWB6 * A0) + (VaWB7 * A3)) + (VaWB8 * A6));
      float Va7 = (((VaWB6 * A1) + (VaWB7 * A4)) + (VaWB8 * A7));
      float Va8 = (((VaWB6 * A2) + (VaWB7 * A5)) + (VaWB8 * A8));
      float VbW0 = world[((j * 12u) + 0u)];
      float VbW1 = world[((j * 12u) + 1u)];
      float VbW2 = world[((j * 12u) + 2u)];
      float VbW3 = world[((j * 12u) + 4u)];
      float VbW4 = world[((j * 12u) + 5u)];
      float VbW5 = world[((j * 12u) + 6u)];
      float VbW6 = world[((j * 12u) + 8u)];
      float VbW7 = world[((j * 12u) + 9u)];
      float VbW8 = world[((j * 12u) + 10u)];
      float VbB0 = bindRot[((j * 9u) + 0u)];
      float VbB1 = bindRot[((j * 9u) + 1u)];
      float VbB2 = bindRot[((j * 9u) + 2u)];
      float VbB3 = bindRot[((j * 9u) + 3u)];
      float VbB4 = bindRot[((j * 9u) + 4u)];
      float VbB5 = bindRot[((j * 9u) + 5u)];
      float VbB6 = bindRot[((j * 9u) + 6u)];
      float VbB7 = bindRot[((j * 9u) + 7u)];
      float VbB8 = bindRot[((j * 9u) + 8u)];
      float VbWB0 = (((VbW0 * VbB0) + (VbW1 * VbB1)) + (VbW2 * VbB2));
      float VbWB1 = (((VbW0 * VbB3) + (VbW1 * VbB4)) + (VbW2 * VbB5));
      float VbWB2 = (((VbW0 * VbB6) + (VbW1 * VbB7)) + (VbW2 * VbB8));
      float VbWB3 = (((VbW3 * VbB0) + (VbW4 * VbB1)) + (VbW5 * VbB2));
      float VbWB4 = (((VbW3 * VbB3) + (VbW4 * VbB4)) + (VbW5 * VbB5));
      float VbWB5 = (((VbW3 * VbB6) + (VbW4 * VbB7)) + (VbW5 * VbB8));
      float VbWB6 = (((VbW6 * VbB0) + (VbW7 * VbB1)) + (VbW8 * VbB2));
      float VbWB7 = (((VbW6 * VbB3) + (VbW7 * VbB4)) + (VbW8 * VbB5));
      float VbWB8 = (((VbW6 * VbB6) + (VbW7 * VbB7)) + (VbW8 * VbB8));
      float Vb0 = (((VbWB0 * A0) + (VbWB1 * A3)) + (VbWB2 * A6));
      float Vb1 = (((VbWB0 * A1) + (VbWB1 * A4)) + (VbWB2 * A7));
      float Vb2 = (((VbWB0 * A2) + (VbWB1 * A5)) + (VbWB2 * A8));
      float Vb3 = (((VbWB3 * A0) + (VbWB4 * A3)) + (VbWB5 * A6));
      float Vb4 = (((VbWB3 * A1) + (VbWB4 * A4)) + (VbWB5 * A7));
      float Vb5 = (((VbWB3 * A2) + (VbWB4 * A5)) + (VbWB5 * A8));
      float Vb6 = (((VbWB6 * A0) + (VbWB7 * A3)) + (VbWB8 * A6));
      float Vb7 = (((VbWB6 * A1) + (VbWB7 * A4)) + (VbWB8 * A7));
      float Vb8 = (((VbWB6 * A2) + (VbWB7 * A5)) + (VbWB8 * A8));
      float Rel0 = (((Va0 * Vb0) + (Va3 * Vb3)) + (Va6 * Vb6));
      float Rel1 = (((Va0 * Vb1) + (Va3 * Vb4)) + (Va6 * Vb7));
      float Rel2 = (((Va0 * Vb2) + (Va3 * Vb5)) + (Va6 * Vb8));
      float Rel3 = (((Va1 * Vb0) + (Va4 * Vb3)) + (Va7 * Vb6));
      float Rel4 = (((Va1 * Vb1) + (Va4 * Vb4)) + (Va7 * Vb7));
      float Rel5 = (((Va1 * Vb2) + (Va4 * Vb5)) + (Va7 * Vb8));
      float Rel6 = (((Va2 * Vb0) + (Va5 * Vb3)) + (Va8 * Vb6));
      float Rel7 = (((Va2 * Vb1) + (Va5 * Vb4)) + (Va8 * Vb7));
      float Rel8 = (((Va2 * Vb2) + (Va5 * Vb5)) + (Va8 * Vb8));
      float tw_t = max((((1.000000 + Rel0) + Rel4) + Rel8), 0.000000);
      float tw = (4.000000 * atan2(((0u == 0u) ? (Rel7 - Rel5) : ((0u == 1u) ? (Rel2 - Rel6) : (Rel3 - Rel1))), (tw_t + (2.000000 * sqrt(tw_t)))));
      float st_v00 = ((0u == 0u) ? 1.000000 : 0.000000);
      float st_v01 = ((0u == 1u) ? 1.000000 : 0.000000);
      float st_v02 = ((0u == 2u) ? 1.000000 : 0.000000);
      float st_ru0 = (((Rel0 * st_v00) + (Rel1 * st_v01)) + (Rel2 * st_v02));
      float st_ru1 = (((Rel3 * st_v00) + (Rel4 * st_v01)) + (Rel5 * st_v02));
      float st_ru2 = (((Rel6 * st_v00) + (Rel7 * st_v01)) + (Rel8 * st_v02));
      float st_u0_l = sqrt((((st_ru0 * st_ru0) + (st_ru1 * st_ru1)) + (st_ru2 * st_ru2)));
      float st_u00 = ((st_u0_l > 1.0e-12f) ? (st_ru0 / st_u0_l) : st_ru0);
      float st_u01 = ((st_u0_l > 1.0e-12f) ? (st_ru1 / st_u0_l) : st_ru1);
      float st_u02 = ((st_u0_l > 1.0e-12f) ? (st_ru2 / st_u0_l) : st_ru2);
      bool st_fy = (abs(st_v01) < 0.900000);
      float st_ref0 = (st_fy ? 0.000000 : 1.000000);
      float st_ref1 = (st_fy ? 1.000000 : 0.000000);
      float st_ref2 = 0.000000;
      float st_v1r0 = ((st_v01 * st_ref2) - (st_v02 * st_ref1));
      float st_v1r1 = ((st_v02 * st_ref0) - (st_v00 * st_ref2));
      float st_v1r2 = ((st_v00 * st_ref1) - (st_v01 * st_ref0));
      float st_v1_l = sqrt((((st_v1r0 * st_v1r0) + (st_v1r1 * st_v1r1)) + (st_v1r2 * st_v1r2)));
      float st_v10 = ((st_v1_l > 1.0e-12f) ? (st_v1r0 / st_v1_l) : st_v1r0);
      float st_v11 = ((st_v1_l > 1.0e-12f) ? (st_v1r1 / st_v1_l) : st_v1r1);
      float st_v12 = ((st_v1_l > 1.0e-12f) ? (st_v1r2 / st_v1_l) : st_v1r2);
      float st_v2r0 = ((st_v01 * st_v12) - (st_v02 * st_v11));
      float st_v2r1 = ((st_v02 * st_v10) - (st_v00 * st_v12));
      float st_v2r2 = ((st_v00 * st_v11) - (st_v01 * st_v10));
      float st_v2_l = sqrt((((st_v2r0 * st_v2r0) + (st_v2r1 * st_v2r1)) + (st_v2r2 * st_v2r2)));
      float st_v20 = ((st_v2_l > 1.0e-12f) ? (st_v2r0 / st_v2_l) : st_v2r0);
      float st_v21 = ((st_v2_l > 1.0e-12f) ? (st_v2r1 / st_v2_l) : st_v2r1);
      float st_v22 = ((st_v2_l > 1.0e-12f) ? (st_v2r2 / st_v2_l) : st_v2r2);
      float st_ax0 = ((st_v01 * st_u02) - (st_v02 * st_u01));
      float st_ax1 = ((st_v02 * st_u00) - (st_v00 * st_u02));
      float st_ax2 = ((st_v00 * st_u01) - (st_v01 * st_u00));
      float st_sn = sqrt((((st_ax0 * st_ax0) + (st_ax1 * st_ax1)) + (st_ax2 * st_ax2)));
      float st_cs = (((st_v00 * st_u00) + (st_v01 * st_u01)) + (st_v02 * st_u02));
      float st_k0 = (st_ax0 / max(st_sn, 1.0e-9f));
      float st_k1 = (st_ax1 / max(st_sn, 1.0e-9f));
      float st_k2 = (st_ax2 / max(st_sn, 1.0e-9f));
      float st_c1_kx0 = ((st_k1 * st_v12) - (st_k2 * st_v11));
      float st_c1_kx1 = ((st_k2 * st_v10) - (st_k0 * st_v12));
      float st_c1_kx2 = ((st_k0 * st_v11) - (st_k1 * st_v10));
      float st_c1_kw = (((st_k0 * st_v10) + (st_k1 * st_v11)) + (st_k2 * st_v12));
      float st_c10 = (((st_cs * st_v10) + (st_sn * st_c1_kx0)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k0));
      float st_c11 = (((st_cs * st_v11) + (st_sn * st_c1_kx1)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k1));
      float st_c12 = (((st_cs * st_v12) + (st_sn * st_c1_kx2)) + ((st_c1_kw * (1.000000 - st_cs)) * st_k2));
      float st_c2_kx0 = ((st_k1 * st_v22) - (st_k2 * st_v21));
      float st_c2_kx1 = ((st_k2 * st_v20) - (st_k0 * st_v22));
      float st_c2_kx2 = ((st_k0 * st_v21) - (st_k1 * st_v20));
      float st_c2_kw = (((st_k0 * st_v20) + (st_k1 * st_v21)) + (st_k2 * st_v22));
      float st_c20 = (((st_cs * st_v20) + (st_sn * st_c2_kx0)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k0));
      float st_c21 = (((st_cs * st_v21) + (st_sn * st_c2_kx1)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k1));
      float st_c22 = (((st_cs * st_v22) + (st_sn * st_c2_kx2)) + ((st_c2_kw * (1.000000 - st_cs)) * st_k2));
      if ((st_sn < 1.0e-9f)) {
        if ((st_cs > 0.000000)) {
          st_c10 = st_v10;
          st_c11 = st_v11;
          st_c12 = st_v12;
          st_c20 = st_v20;
          st_c21 = st_v21;
          st_c22 = st_v22;
        } else {
          bool st_px = ((abs(st_u00) <= abs(st_u01)) && (abs(st_u00) <= abs(st_u02)));
          float st_pr0 = (st_px ? 1.000000 : 0.000000);
          float st_pr1 = (st_px ? 0.000000 : 1.000000);
          float st_pr2 = 0.000000;
          float st_p10 = ((st_u01 * st_pr2) - (st_u02 * st_pr1));
          float st_p11 = ((st_u02 * st_pr0) - (st_u00 * st_pr2));
          float st_p12 = ((st_u00 * st_pr1) - (st_u01 * st_pr0));
          float st_p1n_l = sqrt((((st_p10 * st_p10) + (st_p11 * st_p11)) + (st_p12 * st_p12)));
          float st_p1n0 = ((st_p1n_l > 1.0e-12f) ? (st_p10 / st_p1n_l) : st_p10);
          float st_p1n1 = ((st_p1n_l > 1.0e-12f) ? (st_p11 / st_p1n_l) : st_p11);
          float st_p1n2 = ((st_p1n_l > 1.0e-12f) ? (st_p12 / st_p1n_l) : st_p12);
          float st_p20 = ((st_u01 * st_p1n2) - (st_u02 * st_p1n1));
          float st_p21 = ((st_u02 * st_p1n0) - (st_u00 * st_p1n2));
          float st_p22 = ((st_u00 * st_p1n1) - (st_u01 * st_p1n0));
          float st_p2n_l = sqrt((((st_p20 * st_p20) + (st_p21 * st_p21)) + (st_p22 * st_p22)));
          float st_p2n0 = ((st_p2n_l > 1.0e-12f) ? (st_p20 / st_p2n_l) : st_p20);
          float st_p2n1 = ((st_p2n_l > 1.0e-12f) ? (st_p21 / st_p2n_l) : st_p21);
          float st_p2n2 = ((st_p2n_l > 1.0e-12f) ? (st_p22 / st_p2n_l) : st_p22);
          st_c10 = st_p1n0;
          st_c11 = st_p1n1;
          st_c12 = st_p1n2;
          st_c20 = st_p2n0;
          st_c21 = st_p2n1;
          st_c22 = st_p2n2;
        }
      }
      float st_R00 = (((st_u00 * st_v00) + (st_c10 * st_v10)) + (st_c20 * st_v20));
      float st_R01 = (((st_u00 * st_v01) + (st_c10 * st_v11)) + (st_c20 * st_v21));
      float st_R02 = (((st_u00 * st_v02) + (st_c10 * st_v12)) + (st_c20 * st_v22));
      float st_R03 = (((st_u01 * st_v00) + (st_c11 * st_v10)) + (st_c21 * st_v20));
      float st_R04 = (((st_u01 * st_v01) + (st_c11 * st_v11)) + (st_c21 * st_v21));
      float st_R05 = (((st_u01 * st_v02) + (st_c11 * st_v12)) + (st_c21 * st_v22));
      float st_R06 = (((st_u02 * st_v00) + (st_c12 * st_v10)) + (st_c22 * st_v20));
      float st_R07 = (((st_u02 * st_v01) + (st_c12 * st_v11)) + (st_c22 * st_v21));
      float st_R08 = (((st_u02 * st_v02) + (st_c12 * st_v12)) + (st_c22 * st_v22));
      float st_T0 = (((st_R00 * Rel0) + (st_R03 * Rel3)) + (st_R06 * Rel6));
      float st_T1 = (((st_R00 * Rel1) + (st_R03 * Rel4)) + (st_R06 * Rel7));
      float st_T2 = (((st_R00 * Rel2) + (st_R03 * Rel5)) + (st_R06 * Rel8));
      float st_T3 = (((st_R01 * Rel0) + (st_R04 * Rel3)) + (st_R07 * Rel6));
      float st_T4 = (((st_R01 * Rel1) + (st_R04 * Rel4)) + (st_R07 * Rel7));
      float st_T5 = (((st_R01 * Rel2) + (st_R04 * Rel5)) + (st_R07 * Rel8));
      float st_T6 = (((st_R02 * Rel0) + (st_R05 * Rel3)) + (st_R08 * Rel6));
      float st_T7 = (((st_R02 * Rel1) + (st_R05 * Rel4)) + (st_R08 * Rel7));
      float st_T8 = (((st_R02 * Rel2) + (st_R05 * Rel5)) + (st_R08 * Rel8));
      float st = atan2(((0u == 0u) ? (st_T7 - st_T5) : ((0u == 1u) ? (st_T2 - st_T6) : (st_T3 - st_T1))), ((0u == 0u) ? (st_T4 + st_T8) : ((0u == 1u) ? (st_T8 + st_T0) : (st_T0 + st_T4))));
      al = ((params.twistForm == 0u) ? st : tw);
    }
  }
  chan[((2u * params.NJ) + j)] = al;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaTwist
