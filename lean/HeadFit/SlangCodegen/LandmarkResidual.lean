import Drape.SlangCodegen.Dsl

/-!
# `HeadFit.SlangCodegen.LandmarkResidual` — sparse landmarks, one thread per pair

For landmark `k`, the model vertex `mid[k]` (a region index) and the
avatar point `tgt[k]`:

    d = pos[mid[k]] − tgt[k]
    e[k] = lw[k] · |d|²
    dpos[mid[k]] += 2 · lw[k] · d

It runs after `hf_surface_residual` and adds into that kernel's `dpos`.
The driver refuses a landmark set that names one model vertex twice
(`hf_marks_create`), so no two threads write the same element.

Bindings (set 0):

  0  ConstantBuffer<HfMarkParams> { uint K; }
  1  StructuredBuffer<float>   pos   (P·3)
  2  StructuredBuffer<uint>    mid   (K)
  3  StructuredBuffer<float>   tgt   (K·3)
  4  StructuredBuffer<float>   lw    (K)
  5  RWStructuredBuffer<float> e     (K)
  6  RWStructuredBuffer<float> dpos  (P·3)
-/

namespace HeadFit.SlangCodegen.LandmarkResidual

open LeanSlang
open Drape.SlangCodegen.Dsl

def shader : SlangShaderModule :=
  { structs := [ { name := "HfMarkParams", fields := [fld "K" uT] } ]
  , globals :=
      [ paramsCB "HfMarkParams", roF "pos" 1, roU "mid" 2, roF "tgt" 3, roF "lw" 4, rwF "e" 5, rwF "dpos" 6 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "k" (.member (v "tid") "x")
          , if_ (ge (v "k") (p "K")) [ ret ]
          , let_ uT "o" (at_ "mid" (v "k") * u 3)
          , let_ fT "dx" (at_ "pos" (v "o") - at_ "tgt" (v "k" * u 3))
          , let_ fT "dy" (at_ "pos" (v "o" + u 1) - at_ "tgt" (v "k" * u 3 + u 1))
          , let_ fT "dz" (at_ "pos" (v "o" + u 2) - at_ "tgt" (v "k" * u 3 + u 2))
          , let_ fT "w" (at_ "lw" (v "k"))
          , setAt "e" (v "k") (v "w" * (v "dx" * v "dx" + v "dy" * v "dy" + v "dz" * v "dz"))
          , let_ fT "b" (fl 2.0 * v "w")
          , setAt "dpos" (v "o") (at_ "dpos" (v "o") + v "b" * v "dx")
          , setAt "dpos" (v "o" + u 1) (at_ "dpos" (v "o" + u 1) + v "b" * v "dy")
          , setAt "dpos" (v "o" + u 2) (at_ "dpos" (v "o" + u 2) + v "b" * v "dz") ] ] }

-- BEGIN PIN
def expected : String :=
"struct HfMarkParams {
  uint K;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfMarkParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> pos;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> mid;
[[vk::binding(3, 0)]]
StructuredBuffer<float> tgt;
[[vk::binding(4, 0)]]
StructuredBuffer<float> lw;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> dpos;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint k = tid.x;
  if ((k >= params.K)) {
    return;
  }
  uint o = (mid[k] * 3u);
  float dx = (pos[o] - tgt[(k * 3u)]);
  float dy = (pos[(o + 1u)] - tgt[((k * 3u) + 1u)]);
  float dz = (pos[(o + 2u)] - tgt[((k * 3u) + 2u)]);
  float w = lw[k];
  e[k] = (w * (((dx * dx) + (dy * dy)) + (dz * dz)));
  float b = (2.000000 * w);
  dpos[o] = (dpos[o] + (b * dx));
  dpos[(o + 1u)] = (dpos[(o + 1u)] + (b * dy));
  dpos[(o + 2u)] = (dpos[(o + 2u)] + (b * dz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.LandmarkResidual
