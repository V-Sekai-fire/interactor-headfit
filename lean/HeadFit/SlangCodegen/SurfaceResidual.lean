import Drape.SlangCodegen.Dsl

/-!
# `HeadFit.SlangCodegen.SurfaceResidual` — point-to-surface, one thread per vertex

For each model vertex `pos[i]` with its closest point `q[i]` on the target
face and the target's normal `n[i]` there (found by the driver's BVH for
the current correspondence round, held fixed within it):

    d = pos[i] − q[i]
    e[i]    = wp[i] · (n[i]·d)² + wq[i] · |d|²
    dpos[i] = 2 · (wp[i] · (n[i]·d) · n[i] + wq[i] · d)

The plane term is the distance to the surface's tangent plane at `q`, the
point term keeps the pair from sliding. The weights carry the region
masks (eyes 0, the nose down-weighted) and the normalisation; a rejected
correspondence (a miss, a normal facing away, the target's boundary) has
both weights 0.

Bindings (set 0):

  0  ConstantBuffer<HfSurfParams> { uint P; }
  1  StructuredBuffer<float>   pos   (P·3)
  2  StructuredBuffer<float>   q     (P·3)
  3  StructuredBuffer<float>   n     (P·3)
  4  StructuredBuffer<float>   w     (P·2: wp, wq)
  5  RWStructuredBuffer<float> e     (P)
  6  RWStructuredBuffer<float> dpos  (P·3)
-/

namespace HeadFit.SlangCodegen.SurfaceResidual

open LeanSlang
open Drape.SlangCodegen.Dsl

private def c (k : Nat) : E := v "i" * u 3 + u k

def shader : SlangShaderModule :=
  { structs := [ { name := "HfSurfParams", fields := [fld "P" uT] } ]
  , globals :=
      [ paramsCB "HfSurfParams", roF "pos" 1, roF "q" 2, roF "n" 3, roF "w" 4, rwF "e" 5, rwF "dpos" 6 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "P")) [ ret ]
          , let_ fT "dx" (at_ "pos" (c 0) - at_ "q" (c 0))
          , let_ fT "dy" (at_ "pos" (c 1) - at_ "q" (c 1))
          , let_ fT "dz" (at_ "pos" (c 2) - at_ "q" (c 2))
          , let_ fT "nx" (at_ "n" (c 0))
          , let_ fT "ny" (at_ "n" (c 1))
          , let_ fT "nz" (at_ "n" (c 2))
          , let_ fT "wp" (at_ "w" (v "i" * u 2))
          , let_ fT "wq" (at_ "w" (v "i" * u 2 + u 1))
          , let_ fT "pl" (v "nx" * v "dx" + v "ny" * v "dy" + v "nz" * v "dz")
          , let_ fT "dd" (v "dx" * v "dx" + v "dy" * v "dy" + v "dz" * v "dz")
          , setAt "e" (v "i") (v "wp" * v "pl" * v "pl" + v "wq" * v "dd")
          , let_ fT "a" (fl 2.0 * v "wp" * v "pl")
          , let_ fT "b" (fl 2.0 * v "wq")
          , setAt "dpos" (c 0) (v "a" * v "nx" + v "b" * v "dx")
          , setAt "dpos" (c 1) (v "a" * v "ny" + v "b" * v "dy")
          , setAt "dpos" (c 2) (v "a" * v "nz" + v "b" * v "dz") ] ] }

-- BEGIN PIN
def expected : String :=
"struct HfSurfParams {
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfSurfParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> pos;
[[vk::binding(2, 0)]]
StructuredBuffer<float> q;
[[vk::binding(3, 0)]]
StructuredBuffer<float> n;
[[vk::binding(4, 0)]]
StructuredBuffer<float> w;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> dpos;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  float dx = (pos[((i * 3u) + 0u)] - q[((i * 3u) + 0u)]);
  float dy = (pos[((i * 3u) + 1u)] - q[((i * 3u) + 1u)]);
  float dz = (pos[((i * 3u) + 2u)] - q[((i * 3u) + 2u)]);
  float nx = n[((i * 3u) + 0u)];
  float ny = n[((i * 3u) + 1u)];
  float nz = n[((i * 3u) + 2u)];
  float wp = w[(i * 2u)];
  float wq = w[((i * 2u) + 1u)];
  float pl = (((nx * dx) + (ny * dy)) + (nz * dz));
  float dd = (((dx * dx) + (dy * dy)) + (dz * dz));
  e[i] = (((wp * pl) * pl) + (wq * dd));
  float a = ((2.000000 * wp) * pl);
  float b = (2.000000 * wq);
  dpos[((i * 3u) + 0u)] = ((a * nx) + (b * dx));
  dpos[((i * 3u) + 1u)] = ((a * ny) + (b * dy));
  dpos[((i * 3u) + 2u)] = ((a * nz) + (b * dz));
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end HeadFit.SlangCodegen.SurfaceResidual
