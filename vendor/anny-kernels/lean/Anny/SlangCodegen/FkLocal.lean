import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.FkLocal` — forward kinematics from parent-local 3×4 transforms

    world_j = world_p · local_j        (world_j = local_j for a root, parents[j] = 0xFFFFFFFF)

For any skeleton given as parent-local matrices (an imported avatar's, where `anny_fk` needs
ANNY's bind frames). A transform is 12 floats: the row-major 3×3, then the translation.
`parents[j] < j` (topological order; the host checks it). One thread walks the tree.

Bindings (set 0):

  0  ConstantBuffer<AnnyFkLocalParams> { uint J; }
  1  StructuredBuffer<uint>    parents (J)
  2  StructuredBuffer<float>   local   (J·12)
  3  RWStructuredBuffer<float> world   (J·12)
-/

namespace Anny.SlangCodegen.FkLocal

open LeanSlang
open Drape.SlangCodegen.Dsl

def lc (k : Nat) : E := at_ "local" (v "b" + u k)
def pw (k : Nat) : E := at_ "world" (v "p" + u k)

/-- Row r of world_p times column c of local_j (c = 3 adds world_p's translation). -/
def rowCol (r c : Nat) : E :=
  pw (3 * r) * lc c + pw (3 * r + 1) * lc (3 + c) + pw (3 * r + 2) * lc (6 + c)

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyFkLocalParams", fields := [fld "J" uT] } ]
  , globals := [ paramsCB "AnnyFkLocalParams", roU "parents" 1, roF "local" 2, rwF "world" 3 ]
  , functions :=
      [ entry 1 [dtid]
          [ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ]
          , for_ "j" (u 0) (p "J")
              [ let_ uT "b" (v "j" * u 12)
              , let_ uT "par" (at_ "parents" (v "j"))
              , if_ (eq (v "par") (u 4294967295))
                  [ for_ "k" (u 0) (u 12) [ setAt "world" (v "b" + v "k") (at_ "local" (v "b" + v "k")) ] ]
                  [ let_ uT "p" (v "par" * u 12)
                  , setAt "world" (v "b" + u 0) (rowCol 0 0)
                  , setAt "world" (v "b" + u 1) (rowCol 0 1)
                  , setAt "world" (v "b" + u 2) (rowCol 0 2)
                  , setAt "world" (v "b" + u 3) (rowCol 1 0)
                  , setAt "world" (v "b" + u 4) (rowCol 1 1)
                  , setAt "world" (v "b" + u 5) (rowCol 1 2)
                  , setAt "world" (v "b" + u 6) (rowCol 2 0)
                  , setAt "world" (v "b" + u 7) (rowCol 2 1)
                  , setAt "world" (v "b" + u 8) (rowCol 2 2)
                  , setAt "world" (v "b" + u 9)
                      (pw 0 * lc 9 + pw 1 * lc 10 + pw 2 * lc 11 + pw 9)
                  , setAt "world" (v "b" + u 10)
                      (pw 3 * lc 9 + pw 4 * lc 10 + pw 5 * lc 11 + pw 10)
                  , setAt "world" (v "b" + u 11)
                      (pw 6 * lc 9 + pw 7 * lc 10 + pw 8 * lc 11 + pw 11) ] ] ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyFkLocalParams {
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyFkLocalParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> parents;
[[vk::binding(2, 0)]]
StructuredBuffer<float> local;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> world;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  for (uint j = 0u; j < params.J; ++j) {
    uint b = (j * 12u);
    uint par = parents[j];
    if ((par == 4294967295u)) {
      for (uint k = 0u; k < 12u; ++k) {
        world[(b + k)] = local[(b + k)];
      }
    } else {
      uint p = (par * 12u);
      world[(b + 0u)] = (((world[(p + 0u)] * local[(b + 0u)]) + (world[(p + 1u)] * local[(b + 3u)])) + (world[(p + 2u)] * local[(b + 6u)]));
      world[(b + 1u)] = (((world[(p + 0u)] * local[(b + 1u)]) + (world[(p + 1u)] * local[(b + 4u)])) + (world[(p + 2u)] * local[(b + 7u)]));
      world[(b + 2u)] = (((world[(p + 0u)] * local[(b + 2u)]) + (world[(p + 1u)] * local[(b + 5u)])) + (world[(p + 2u)] * local[(b + 8u)]));
      world[(b + 3u)] = (((world[(p + 3u)] * local[(b + 0u)]) + (world[(p + 4u)] * local[(b + 3u)])) + (world[(p + 5u)] * local[(b + 6u)]));
      world[(b + 4u)] = (((world[(p + 3u)] * local[(b + 1u)]) + (world[(p + 4u)] * local[(b + 4u)])) + (world[(p + 5u)] * local[(b + 7u)]));
      world[(b + 5u)] = (((world[(p + 3u)] * local[(b + 2u)]) + (world[(p + 4u)] * local[(b + 5u)])) + (world[(p + 5u)] * local[(b + 8u)]));
      world[(b + 6u)] = (((world[(p + 6u)] * local[(b + 0u)]) + (world[(p + 7u)] * local[(b + 3u)])) + (world[(p + 8u)] * local[(b + 6u)]));
      world[(b + 7u)] = (((world[(p + 6u)] * local[(b + 1u)]) + (world[(p + 7u)] * local[(b + 4u)])) + (world[(p + 8u)] * local[(b + 7u)]));
      world[(b + 8u)] = (((world[(p + 6u)] * local[(b + 2u)]) + (world[(p + 7u)] * local[(b + 5u)])) + (world[(p + 8u)] * local[(b + 8u)]));
      world[(b + 9u)] = ((((world[(p + 0u)] * local[(b + 9u)]) + (world[(p + 1u)] * local[(b + 10u)])) + (world[(p + 2u)] * local[(b + 11u)])) + world[(p + 9u)]);
      world[(b + 10u)] = ((((world[(p + 3u)] * local[(b + 9u)]) + (world[(p + 4u)] * local[(b + 10u)])) + (world[(p + 5u)] * local[(b + 11u)])) + world[(p + 10u)]);
      world[(b + 11u)] = ((((world[(p + 6u)] * local[(b + 9u)]) + (world[(p + 7u)] * local[(b + 10u)])) + (world[(p + 8u)] * local[(b + 11u)])) + world[(p + 11u)]);
    }
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.FkLocal
