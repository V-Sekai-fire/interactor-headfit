import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.SomaCorrL1` — the correctives' hidden layer, one thread per hidden unit

    z_k = act(Σ_d feat_d · W1[d, k]),   act = relu, then tanh if useTanh

`CorrectivesMLP.forward`'s first layer: `feat` is `soma_orient`'s 6 features
per joint (D = 6·NJ), `W1` the checkpoint's weights already multiplied by
`M1_prior` on the host (row-major D × K). `useTanh` is the checkpoint's
`use_tanh` (0 or 1).

Bindings (set 0):

  0  ConstantBuffer<SomaCorrL1Params> { uint D; uint K; uint useTanh; }
  1  StructuredBuffer<float>   feat  (D)
  2  StructuredBuffer<float>   w1    (D·K)
  3  RWStructuredBuffer<float> z     (K)
-/

namespace Anny.SlangCodegen.SomaCorrL1

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "SomaCorrL1Params", fields := [fld "D" uT, fld "K" uT, fld "useTanh" uT] } ]
  , globals := [ paramsCB "SomaCorrL1Params", roF "feat" 1, roF "w1" 2, rwF "z" 3 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "k" (.member (v "tid") "x")
          , if_ (ge (v "k") (p "K")) [ ret ]
          , let_ fT "a" (fl 0.0)
          , for_ "d" (u 0) (p "D")
              [ setv "a" (v "a" + at_ "feat" (v "d") * at_ "w1" (v "d" * p "K" + v "k")) ]
          , setv "a" (fmax (v "a") (fl 0.0))
          , if_ (ne (p "useTanh") (u 0)) [ setv "a" (call "tanh" [v "a"]) ]
          , setAt "z" (v "k") (v "a") ] ] }

-- BEGIN PIN
def expected : String :=
"struct SomaCorrL1Params {
  uint D;
  uint K;
  uint useTanh;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaCorrL1Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> feat;
[[vk::binding(2, 0)]]
StructuredBuffer<float> w1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> z;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint k = tid.x;
  if ((k >= params.K)) {
    return;
  }
  float a = 0.000000;
  for (uint d = 0u; d < params.D; ++d) {
    a = (a + (feat[d] * w1[((d * params.K) + k)]));
  }
  a = max(a, 0.000000);
  if ((params.useTanh != 0u)) {
    a = tanh(a);
  }
  z[k] = a;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.SomaCorrL1
