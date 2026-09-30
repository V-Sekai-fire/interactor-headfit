import Drape.SlangCodegen.Dsl

/-!
# `HeadFit.SlangCodegen.Prior` — the quadratic prior on the unknowns

    e[j] = lam[j] · (x[j] − x0[j])²,   g[j] = 2 · lam[j] · (x[j] − x0[j])

One thread per unknown. It holds the local dials near zero and the
phenotype near ANNY's default, so a surface the fit cannot see (the back
of the head under the hair, the masked eyes) does not wander. `lam` is 0
for the rigid unknowns. `g` is the start of the gradient: the host adds
the chain through the blend and the similarity into it.

Bindings (set 0):

  0  ConstantBuffer<HfPriorParams> { uint N; }
  1  StructuredBuffer<float>   x    (N)
  2  StructuredBuffer<float>   x0   (N)
  3  StructuredBuffer<float>   lam  (N)
  4  RWStructuredBuffer<float> e    (N)
  5  RWStructuredBuffer<float> g    (N)
-/

namespace HeadFit.SlangCodegen.Prior

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "HfPriorParams", fields := [fld "N" uT] } ]
  , globals := [ paramsCB "HfPriorParams", roF "x" 1, roF "x0" 2, roF "lam" 3, rwF "e" 4, rwF "g" 5 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "j" (.member (v "tid") "x")
          , if_ (ge (v "j") (p "N")) [ ret ]
          , let_ fT "d" (at_ "x" (v "j") - at_ "x0" (v "j"))
          , let_ fT "l" (at_ "lam" (v "j"))
          , setAt "e" (v "j") (v "l" * v "d" * v "d")
          , setAt "g" (v "j") (fl 2.0 * v "l" * v "d") ] ] }

-- BEGIN PIN
def expected : String :=
"struct HfPriorParams {
  uint N;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfPriorParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x;
[[vk::binding(2, 0)]]
StructuredBuffer<float> x0;
[[vk::binding(3, 0)]]
StructuredBuffer<float> lam;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> g;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.N)) {
    return;
  }
  float d = (x[j] - x0[j]);
  float l = lam[j];
  e[j] = ((l * d) * d);
  g[j] = ((2.000000 * l) * d);
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.Prior
