import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common
import Anny.SlangCodegen.SomaCommon

/-!
# `Anny.SlangCodegen.SomaExpand` — the target rig's seeds, one thread per target joint

`SOMAProceduralParameterTransform.expand_world_transforms_from_source_fk`,
per target joint t by its `kind`:

    0 control  acc_t = publicWorld[refIdx_t]                       anc_t = none
    1 twist    a     = sign_r · Σ_mode Σ_j M[mode, r, j] · chan[mode, j]   (r = refIdx_t)
               L     = Base_t · Rot(axis_r, a)
               acc_t = publicWorld[par_t] · [L | localT_t]          anc_t = none
    2 fill     acc_t = [Base_t | localT_t]                          anc_t = par_t

`Rot(axis, a)` is the single-axis 3×3 built from cos a and sin a
(`single_axis_rotation_matrices`). `Base`/`localT` are the target rig's
bind-local rotations and translations; for a twist joint `par_t` is the
public index of its parent, which the exporter checks is a control joint
(SOMA-X reads the parent's world before any fill). The mode sums are
taken one mode at a time and then added, as SOMA-X adds `values @ Mᵀ`
per mode. `soma_jump` rounds over the target tree then finish the fill
joints, whose ancestors start at a joint with no ancestor left.

Bindings (set 0):

   0  ConstantBuffer<SomaExpandParams> { uint NT; uint NJ; uint NTW; }
   1  StructuredBuffer<uint>    kind        (NT)
   2  StructuredBuffer<uint>    refIdx      (NT)
   3  StructuredBuffer<uint>    par         (NT)
   4  StructuredBuffer<float>   baseRot     (NT·9)
   5  StructuredBuffer<float>   localT      (NT·3)
   6  StructuredBuffer<float>   twistM      (3·NTW·NJ)
   7  StructuredBuffer<float>   chan        (3·NJ)
   8  StructuredBuffer<uint>    twAxis      (NTW)
   9  StructuredBuffer<float>   twSign      (NTW)
  10  StructuredBuffer<float>   publicWorld (NJ·12)
  11  RWStructuredBuffer<float> acc         (NT·12)
  12  RWStructuredBuffer<uint>  anc         (NT)
-/

namespace Anny.SlangCodegen.SomaExpand

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common
open Anny.SlangCodegen.SomaCommon

/-- Entry (i, j) of the rotation by `c`, `sn` about the runtime axis `ax`. -/
def axisRot (i j : Nat) : E :=
  let c := v "c"
  let sn := v "sn"
  let z := fl 0.0
  let o := fl 1.0
  -- about x: [1 0 0; 0 c −s; 0 s c]
  let rx : List E := [o, z, z, z, c, -sn, z, sn, c]
  -- about y: [c 0 s; 0 1 0; −s 0 c]
  let ry : List E := [c, z, sn, z, o, z, -sn, z, c]
  -- about z: [c −s 0; s c 0; 0 0 1]
  let rz : List E := [c, -sn, z, sn, c, z, z, z, o]
  let k := 3 * i + j
  sel (eq (v "ax") (u 0)) (rx.getD k z) (sel (eq (v "ax") (u 1)) (ry.getD k z) (rz.getD k z))

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaExpandParams", fields := [fld "NT" uT, fld "NJ" uT, fld "NTW" uT] } ]
  , globals :=
      [ paramsCB "SomaExpandParams", roU "kind" 1, roU "refIdx" 2, roU "par" 3, roF "baseRot" 4
      , roF "localT" 5, roF "twistM" 6, roF "chan" 7, roU "twAxis" 8, roF "twSign" 9
      , roF "publicWorld" 10, rwF "acc" 11, rwU "anc" 12 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "t" (.member (v "tid") "x")
           , if_ (ge (v "t") (p "NT")) [ ret ]
           , let_ uT "k" (at_ "kind" (v "t"))
           , let_ uT "r" (at_ "refIdx" (v "t"))
           , let_ uT "ob" (v "t" * u 12) ] ++
           [ if_ (eq (v "k") (u 0))
               ([ let_ uT "os" (v "r" * u 12)
                , for_ "q" (u 0) (u 12)
                    [ setAt "acc" (v "ob" + v "q") (at_ "publicWorld" (v "os" + v "q")) ]
                , setAt "anc" (v "t") (u 4294967295) ])
               (load9 "Bs" "baseRot" (v "t" * u 9) ++
                vec "lt" (fun i => at_ "localT" (v "t" * u 3 + u i)) ++
                [ if_ (eq (v "k") (u 1))
                    ([ let_ fT "ang" (fl 0.0)
                     , for_ "md" (u 0) (u 3)
                         [ let_ fT "sm" (fl 0.0)
                         , let_ uT "mo" ((v "md" * p "NTW" + v "r") * p "NJ")
                         , for_ "jj" (u 0) (p "NJ")
                             [ setv "sm" (v "sm" + at_ "twistM" (v "mo" + v "jj") * at_ "chan" (v "md" * p "NJ" + v "jj")) ]
                         , setv "ang" (v "ang" + v "sm") ]
                     , let_ fT "a" (v "ang" * at_ "twSign" (v "r"))
                     , let_ fT "c" (call "cos" [v "a"])
                     , let_ fT "sn" (call "sin" [v "a"])
                     , let_ uT "ax" (at_ "twAxis" (v "r")) ] ++
                     mat "Ax" axisRot ++
                     matMul "L" "Bs" "Ax" ++
                     [ let_ uT "op" (at_ "par" (v "t") * u 12) ] ++
                     loadRot12 "P" "publicWorld" (v "op") ++ loadT12 "pt" "publicWorld" (v "op") ++
                     matMul "C" "P" "L" ++ matVec "Pl" "P" "lt" ++
                     vec "c3" (fun i => s "Pl" i + s "pt" i) ++
                     store12 "acc" (v "ob") "C" "c3" ++
                     [ setAt "anc" (v "t") (u 4294967295) ])
                    (store12 "acc" (v "ob") "Bs" "lt" ++
                     [ setAt "anc" (v "t") (at_ "par" (v "t")) ]) ]) ]) ] }

-- BEGIN PIN
def expected : String :=
"struct SomaExpandParams {
  uint NT;
  uint NJ;
  uint NTW;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaExpandParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> kind;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> refIdx;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> par;
[[vk::binding(4, 0)]]
StructuredBuffer<float> baseRot;
[[vk::binding(5, 0)]]
StructuredBuffer<float> localT;
[[vk::binding(6, 0)]]
StructuredBuffer<float> twistM;
[[vk::binding(7, 0)]]
StructuredBuffer<float> chan;
[[vk::binding(8, 0)]]
StructuredBuffer<uint> twAxis;
[[vk::binding(9, 0)]]
StructuredBuffer<float> twSign;
[[vk::binding(10, 0)]]
StructuredBuffer<float> publicWorld;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float> acc;
[[vk::binding(12, 0)]]
RWStructuredBuffer<uint> anc;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.NT)) {
    return;
  }
  uint k = kind[t];
  uint r = refIdx[t];
  uint ob = (t * 12u);
  if ((k == 0u)) {
    uint os = (r * 12u);
    for (uint q = 0u; q < 12u; ++q) {
      acc[(ob + q)] = publicWorld[(os + q)];
    }
    anc[t] = 4294967295u;
  } else {
    float Bs0 = baseRot[((t * 9u) + 0u)];
    float Bs1 = baseRot[((t * 9u) + 1u)];
    float Bs2 = baseRot[((t * 9u) + 2u)];
    float Bs3 = baseRot[((t * 9u) + 3u)];
    float Bs4 = baseRot[((t * 9u) + 4u)];
    float Bs5 = baseRot[((t * 9u) + 5u)];
    float Bs6 = baseRot[((t * 9u) + 6u)];
    float Bs7 = baseRot[((t * 9u) + 7u)];
    float Bs8 = baseRot[((t * 9u) + 8u)];
    float lt0 = localT[((t * 3u) + 0u)];
    float lt1 = localT[((t * 3u) + 1u)];
    float lt2 = localT[((t * 3u) + 2u)];
    if ((k == 1u)) {
      float ang = 0.000000;
      for (uint md = 0u; md < 3u; ++md) {
        float sm = 0.000000;
        uint mo = (((md * params.NTW) + r) * params.NJ);
        for (uint jj = 0u; jj < params.NJ; ++jj) {
          sm = (sm + (twistM[(mo + jj)] * chan[((md * params.NJ) + jj)]));
        }
        ang = (ang + sm);
      }
      float a = (ang * twSign[r]);
      float c = cos(a);
      float sn = sin(a);
      uint ax = twAxis[r];
      float Ax0 = ((ax == 0u) ? 1.000000 : ((ax == 1u) ? c : c));
      float Ax1 = ((ax == 0u) ? 0.000000 : ((ax == 1u) ? 0.000000 : (-sn)));
      float Ax2 = ((ax == 0u) ? 0.000000 : ((ax == 1u) ? sn : 0.000000));
      float Ax3 = ((ax == 0u) ? 0.000000 : ((ax == 1u) ? 0.000000 : sn));
      float Ax4 = ((ax == 0u) ? c : ((ax == 1u) ? 1.000000 : c));
      float Ax5 = ((ax == 0u) ? (-sn) : ((ax == 1u) ? 0.000000 : 0.000000));
      float Ax6 = ((ax == 0u) ? 0.000000 : ((ax == 1u) ? (-sn) : 0.000000));
      float Ax7 = ((ax == 0u) ? sn : ((ax == 1u) ? 0.000000 : 0.000000));
      float Ax8 = ((ax == 0u) ? c : ((ax == 1u) ? c : 1.000000));
      float L0 = (((Bs0 * Ax0) + (Bs1 * Ax3)) + (Bs2 * Ax6));
      float L1 = (((Bs0 * Ax1) + (Bs1 * Ax4)) + (Bs2 * Ax7));
      float L2 = (((Bs0 * Ax2) + (Bs1 * Ax5)) + (Bs2 * Ax8));
      float L3 = (((Bs3 * Ax0) + (Bs4 * Ax3)) + (Bs5 * Ax6));
      float L4 = (((Bs3 * Ax1) + (Bs4 * Ax4)) + (Bs5 * Ax7));
      float L5 = (((Bs3 * Ax2) + (Bs4 * Ax5)) + (Bs5 * Ax8));
      float L6 = (((Bs6 * Ax0) + (Bs7 * Ax3)) + (Bs8 * Ax6));
      float L7 = (((Bs6 * Ax1) + (Bs7 * Ax4)) + (Bs8 * Ax7));
      float L8 = (((Bs6 * Ax2) + (Bs7 * Ax5)) + (Bs8 * Ax8));
      uint op = (par[t] * 12u);
      float P0 = publicWorld[(op + 0u)];
      float P1 = publicWorld[(op + 1u)];
      float P2 = publicWorld[(op + 2u)];
      float P3 = publicWorld[(op + 4u)];
      float P4 = publicWorld[(op + 5u)];
      float P5 = publicWorld[(op + 6u)];
      float P6 = publicWorld[(op + 8u)];
      float P7 = publicWorld[(op + 9u)];
      float P8 = publicWorld[(op + 10u)];
      float pt0 = publicWorld[(op + 3u)];
      float pt1 = publicWorld[(op + 7u)];
      float pt2 = publicWorld[(op + 11u)];
      float C0 = (((P0 * L0) + (P1 * L3)) + (P2 * L6));
      float C1 = (((P0 * L1) + (P1 * L4)) + (P2 * L7));
      float C2 = (((P0 * L2) + (P1 * L5)) + (P2 * L8));
      float C3 = (((P3 * L0) + (P4 * L3)) + (P5 * L6));
      float C4 = (((P3 * L1) + (P4 * L4)) + (P5 * L7));
      float C5 = (((P3 * L2) + (P4 * L5)) + (P5 * L8));
      float C6 = (((P6 * L0) + (P7 * L3)) + (P8 * L6));
      float C7 = (((P6 * L1) + (P7 * L4)) + (P8 * L7));
      float C8 = (((P6 * L2) + (P7 * L5)) + (P8 * L8));
      float Pl0 = (((P0 * lt0) + (P1 * lt1)) + (P2 * lt2));
      float Pl1 = (((P3 * lt0) + (P4 * lt1)) + (P5 * lt2));
      float Pl2 = (((P6 * lt0) + (P7 * lt1)) + (P8 * lt2));
      float c30 = (Pl0 + pt0);
      float c31 = (Pl1 + pt1);
      float c32 = (Pl2 + pt2);
      acc[(ob + 0u)] = C0;
      acc[(ob + 1u)] = C1;
      acc[(ob + 2u)] = C2;
      acc[(ob + 4u)] = C3;
      acc[(ob + 5u)] = C4;
      acc[(ob + 6u)] = C5;
      acc[(ob + 8u)] = C6;
      acc[(ob + 9u)] = C7;
      acc[(ob + 10u)] = C8;
      acc[(ob + 3u)] = c30;
      acc[(ob + 7u)] = c31;
      acc[(ob + 11u)] = c32;
      anc[t] = 4294967295u;
    } else {
      acc[(ob + 0u)] = Bs0;
      acc[(ob + 1u)] = Bs1;
      acc[(ob + 2u)] = Bs2;
      acc[(ob + 4u)] = Bs3;
      acc[(ob + 5u)] = Bs4;
      acc[(ob + 6u)] = Bs5;
      acc[(ob + 8u)] = Bs6;
      acc[(ob + 9u)] = Bs7;
      acc[(ob + 10u)] = Bs8;
      acc[(ob + 3u)] = lt0;
      acc[(ob + 7u)] = lt1;
      acc[(ob + 11u)] = lt2;
      anc[t] = par[t];
    }
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaExpand
