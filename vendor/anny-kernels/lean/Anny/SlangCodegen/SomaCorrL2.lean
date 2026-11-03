import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.SomaCorrL2` — the correctives' output layer plus the rest shape, one thread per vertex

    bind_v = rest_v + Σ_{e ∈ row v} z[col_e] · val_e        (val_e a 3-vector)

`CorrectivesMLP.forward`'s `y = z @ W2` (W2 times `M2_prior`, and times the
identity's global scale, on the host) then `rest_shape + out` from
`SOMALayer.pose`. W2 is stored by vertex as CSR: for this identity it is
1872 × 54168 with 2.6 M non-zeros, so the dense form `anny_blend` takes
(405 MB) is not used. `bind` is the rest shape `anny_lbs` skins.

Bindings (set 0):

  0  ConstantBuffer<SomaCorrL2Params> { uint V; }
  1  StructuredBuffer<float>   rest    (V·3)
  2  StructuredBuffer<uint>    rowPtr  (V + 1)
  3  StructuredBuffer<uint>    col     (E)
  4  StructuredBuffer<float>   val     (E·3)
  5  StructuredBuffer<float>   z       (K)
  6  RWStructuredBuffer<float> bind    (V·3)
-/

namespace Anny.SlangCodegen.SomaCorrL2

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaCorrL2Params", fields := [fld "V" uT] } ]
  , globals :=
      [ paramsCB "SomaCorrL2Params", roF "rest" 1, roU "rowPtr" 2, roU "col" 3, roF "val" 4
      , roF "z" 5, rwF "bind" 6 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "V")) [ ret ]
          , let_ fT "x" (fl 0.0)
          , let_ fT "y" (fl 0.0)
          , let_ fT "w" (fl 0.0)
          , for_ "e" (at_ "rowPtr" (v "i")) (at_ "rowPtr" (v "i" + u 1))
              [ let_ fT "zk" (at_ "z" (at_ "col" (v "e")))
              , setv "x" (v "x" + v "zk" * at_ "val" (v "e" * u 3))
              , setv "y" (v "y" + v "zk" * at_ "val" (v "e" * u 3 + u 1))
              , setv "w" (v "w" + v "zk" * at_ "val" (v "e" * u 3 + u 2)) ]
          , setAt "bind" (v "i" * u 3) (at_ "rest" (v "i" * u 3) + v "x")
          , setAt "bind" (v "i" * u 3 + u 1) (at_ "rest" (v "i" * u 3 + u 1) + v "y")
          , setAt "bind" (v "i" * u 3 + u 2) (at_ "rest" (v "i" * u 3 + u 2) + v "w") ] ] }

-- BEGIN PIN
def expected : String :=
"struct SomaCorrL2Params {
  uint V;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaCorrL2Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> rest;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> rowPtr;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(4, 0)]]
StructuredBuffer<float> val;
[[vk::binding(5, 0)]]
StructuredBuffer<float> z;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> bind;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float x = 0.000000;
  float y = 0.000000;
  float w = 0.000000;
  for (uint e = rowPtr[i]; e < rowPtr[(i + 1u)]; ++e) {
    float zk = z[col[e]];
    x = (x + (zk * val[(e * 3u)]));
    y = (y + (zk * val[((e * 3u) + 1u)]));
    w = (w + (zk * val[((e * 3u) + 2u)]));
  }
  bind[(i * 3u)] = (rest[(i * 3u)] + x);
  bind[((i * 3u) + 1u)] = (rest[((i * 3u) + 1u)] + y);
  bind[((i * 3u) + 2u)] = (rest[((i * 3u) + 2u)] + w);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaCorrL2
