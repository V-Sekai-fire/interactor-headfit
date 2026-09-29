import Drape.SlangCodegen.Dsl

/-!
# `HeadFit.SlangCodegen.Blend` — the head's shape, one thread per vertex

    v[i] = base[i] + Σ_c coeffs[c] · blend[c·P + i]

The head region of ANNY's MPFB2 base mesh (P vertices) under the macro
(phenotype) targets and the local head and face dials, as dense rows over
the region. `coeffs` come from the phenotype anchor rule (host scalars,
ported from anny-creator's `anny_coeffs.gd`) and the dial values.

Bindings (set 0):

  0  ConstantBuffer<HfBlendParams> { uint P; uint NB; }
  1  StructuredBuffer<float>   base    (P·3)
  2  StructuredBuffer<float>   blend   (NB·P·3)
  3  StructuredBuffer<float>   coeffs  (NB)
  4  RWStructuredBuffer<float> v       (P·3)
-/

namespace HeadFit.SlangCodegen.Blend

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "HfBlendParams", fields := [fld "P" uT, fld "NB" uT] } ]
  , globals :=
      [ paramsCB "HfBlendParams", roF "base" 1, roF "blend" 2, roF "coeffs" 3, rwF "v" 4 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ fT "x" (at_ "base" (v "i" * u 3))
          , let_ fT "y" (at_ "base" (v "i" * u 3 + u 1))
          , let_ fT "z" (at_ "base" (v "i" * u 3 + u 2))
          , for_ "c" (u 0) (p "NB")
              [ let_ fT "w" (at_ "coeffs" (v "c"))
              , let_ uT "o" ((v "c" * p "P" + v "i") * u 3)
              , setv "x" (v "x" + v "w" * at_ "blend" (v "o"))
              , setv "y" (v "y" + v "w" * at_ "blend" (v "o" + u 1))
              , setv "z" (v "z" + v "w" * at_ "blend" (v "o" + u 2)) ]
          , setAt "v" (v "i" * u 3) (v "x")
          , setAt "v" (v "i" * u 3 + u 1) (v "y")
          , setAt "v" (v "i" * u 3 + u 2) (v "z") ] ] }

-- BEGIN PIN
def expected : String :=
"struct HfBlendParams {
  uint P;
  uint NB;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfBlendParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> base;
[[vk::binding(2, 0)]]
StructuredBuffer<float> blend;
[[vk::binding(3, 0)]]
StructuredBuffer<float> coeffs;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> v;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  float x = base[(i * 3u)];
  float y = base[((i * 3u) + 1u)];
  float z = base[((i * 3u) + 2u)];
  for (uint c = 0u; c < params.NB; ++c) {
    float w = coeffs[c];
    uint o = (((c * params.P) + i) * 3u);
    x = (x + (w * blend[o]));
    y = (y + (w * blend[(o + 1u)]));
    z = (z + (w * blend[(o + 2u)]));
  }
  v[(i * 3u)] = x;
  v[((i * 3u) + 1u)] = y;
  v[((i * 3u) + 2u)] = z;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.Blend
