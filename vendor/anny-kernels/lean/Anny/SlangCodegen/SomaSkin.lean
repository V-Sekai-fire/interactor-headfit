import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common

/-!
# `Anny.SlangCodegen.SomaSkin` — skinning matrices, one thread per target joint

    bone_t = world_t · invBind_t        (BatchedSkinning.linear_blend_skinning)

`invBind` is the inverse of the identity's bind transforms that SOMA-X's
`rebind` keeps; all three are 3×4 affines `[R | t]`, so `bone` is ready for
`anny_lbs`.

Bindings (set 0):

  0  ConstantBuffer<SomaSkinParams> { uint NT; }
  1  StructuredBuffer<float>   world    (NT·12)
  2  StructuredBuffer<float>   invBind  (NT·12)
  3  RWStructuredBuffer<float> bone     (NT·12)
-/

namespace Anny.SlangCodegen.SomaSkin

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaSkinParams", fields := [fld "NT" uT] } ]
  , globals := [ paramsCB "SomaSkinParams", roF "world" 1, roF "invBind" 2, rwF "bone" 3 ]
  , functions :=
      [ entry 64 [dtid]
          ([ let_ uT "t" (.member (v "tid") "x")
           , if_ (ge (v "t") (p "NT")) [ ret ]
           , let_ uT "o" (v "t" * u 12) ] ++
           loadRot12 "W" "world" (v "o") ++ loadT12 "wt" "world" (v "o") ++
           loadRot12 "K" "invBind" (v "o") ++ loadT12 "kt" "invBind" (v "o") ++
           matMul "C" "W" "K" ++ matVec "Wk" "W" "kt" ++
           vec "c" (fun i => s "Wk" i + s "wt" i) ++
           store12 "bone" (v "o") "C" "c") ] }

-- BEGIN PIN
def expected : String :=
"struct SomaSkinParams {
  uint NT;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaSkinParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> world;
[[vk::binding(2, 0)]]
StructuredBuffer<float> invBind;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> bone;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint t = tid.x;
  if ((t >= params.NT)) {
    return;
  }
  uint o = (t * 12u);
  float W0 = world[(o + 0u)];
  float W1 = world[(o + 1u)];
  float W2 = world[(o + 2u)];
  float W3 = world[(o + 4u)];
  float W4 = world[(o + 5u)];
  float W5 = world[(o + 6u)];
  float W6 = world[(o + 8u)];
  float W7 = world[(o + 9u)];
  float W8 = world[(o + 10u)];
  float wt0 = world[(o + 3u)];
  float wt1 = world[(o + 7u)];
  float wt2 = world[(o + 11u)];
  float K0 = invBind[(o + 0u)];
  float K1 = invBind[(o + 1u)];
  float K2 = invBind[(o + 2u)];
  float K3 = invBind[(o + 4u)];
  float K4 = invBind[(o + 5u)];
  float K5 = invBind[(o + 6u)];
  float K6 = invBind[(o + 8u)];
  float K7 = invBind[(o + 9u)];
  float K8 = invBind[(o + 10u)];
  float kt0 = invBind[(o + 3u)];
  float kt1 = invBind[(o + 7u)];
  float kt2 = invBind[(o + 11u)];
  float C0 = (((W0 * K0) + (W1 * K3)) + (W2 * K6));
  float C1 = (((W0 * K1) + (W1 * K4)) + (W2 * K7));
  float C2 = (((W0 * K2) + (W1 * K5)) + (W2 * K8));
  float C3 = (((W3 * K0) + (W4 * K3)) + (W5 * K6));
  float C4 = (((W3 * K1) + (W4 * K4)) + (W5 * K7));
  float C5 = (((W3 * K2) + (W4 * K5)) + (W5 * K8));
  float C6 = (((W6 * K0) + (W7 * K3)) + (W8 * K6));
  float C7 = (((W6 * K1) + (W7 * K4)) + (W8 * K7));
  float C8 = (((W6 * K2) + (W7 * K5)) + (W8 * K8));
  float Wk0 = (((W0 * kt0) + (W1 * kt1)) + (W2 * kt2));
  float Wk1 = (((W3 * kt0) + (W4 * kt1)) + (W5 * kt2));
  float Wk2 = (((W6 * kt0) + (W7 * kt1)) + (W8 * kt2));
  float c0 = (Wk0 + wt0);
  float c1 = (Wk1 + wt1);
  float c2 = (Wk2 + wt2);
  bone[(o + 0u)] = C0;
  bone[(o + 1u)] = C1;
  bone[(o + 2u)] = C2;
  bone[(o + 4u)] = C3;
  bone[(o + 5u)] = C4;
  bone[(o + 6u)] = C5;
  bone[(o + 8u)] = C6;
  bone[(o + 9u)] = C7;
  bone[(o + 10u)] = C8;
  bone[(o + 3u)] = c0;
  bone[(o + 7u)] = c1;
  bone[(o + 11u)] = c2;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaSkin
