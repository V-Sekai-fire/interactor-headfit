import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.CapsuleDepth` — how deep each point sits inside a set of capsules

    depth[v] = max_c ( r_c − | p_v − closest point of segment a_c b_c to p_v | )

One thread per point (Gate 10b: a skinned skirt vertex against the leg capsules). A positive depth
is a penetration; the host takes the maximum. A capsule is seven floats: a, b, r.

Bindings (set 0):

  0  ConstantBuffer<AnnyCapsuleDepthParams> { uint V; uint C; }
  1  StructuredBuffer<float>   points   (V·3)
  2  StructuredBuffer<float>   capsules (C·7)
  3  RWStructuredBuffer<float> depth    (V)
-/

namespace Anny.SlangCodegen.CapsuleDepth

open LeanSlang
open Drape.SlangCodegen.Dsl

/-- `capsules[7c + k]`. -/
def cp (k : Nat) : E := at_ "capsules" (v "b" + u k)

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyCapsuleDepthParams", fields := [fld "V" uT, fld "C" uT] } ]
  , globals := [ paramsCB "AnnyCapsuleDepthParams", roF "points" 1, roF "capsules" 2, rwF "depth" 3 ]
  , functions :=
      [ entry 64 [dtid]
          [ let_ uT "i" (.member (v "tid") "x")
          , if_ (ge (v "i") (p "V")) [ ret ]
          , let_ fT "px" (at_ "points" (v "i" * u 3))
          , let_ fT "py" (at_ "points" (v "i" * u 3 + u 1))
          , let_ fT "pz" (at_ "points" (v "i" * u 3 + u 2))
          , let_ fT "best" (fl 0.0 - fltMax)
          , for_ "c" (u 0) (p "C")
              [ let_ uT "b" (v "c" * u 7)
              , let_ fT "dx" (cp 3 - cp 0)
              , let_ fT "dy" (cp 4 - cp 1)
              , let_ fT "dz" (cp 5 - cp 2)
              , let_ fT "len2" (v "dx" * v "dx" + v "dy" * v "dy" + v "dz" * v "dz")
              , let_ fT "t" (fl 0.0)
              , if_ (gt (v "len2") (fl 0.0))
                  [ setv "t" (fmin (fl 1.0) (fmax (fl 0.0)
                      (((v "px" - cp 0) * v "dx" + (v "py" - cp 1) * v "dy" + (v "pz" - cp 2) * v "dz") / v "len2"))) ]
              , let_ fT "ex" (v "px" - (cp 0 + v "t" * v "dx"))
              , let_ fT "ey" (v "py" - (cp 1 + v "t" * v "dy"))
              , let_ fT "ez" (v "pz" - (cp 2 + v "t" * v "dz"))
              , setv "best" (fmax (v "best") (cp 6 - call "sqrt" [v "ex" * v "ex" + v "ey" * v "ey" + v "ez" * v "ez"])) ]
          , setAt "depth" (v "i") (v "best") ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyCapsuleDepthParams {
  uint V;
  uint C;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyCapsuleDepthParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> points;
[[vk::binding(2, 0)]]
StructuredBuffer<float> capsules;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> depth;

[shader(\"compute\")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float px = points[(i * 3u)];
  float py = points[((i * 3u) + 1u)];
  float pz = points[((i * 3u) + 2u)];
  float best = (0.000000 - asfloat(2139095039u));
  for (uint c = 0u; c < params.C; ++c) {
    uint b = (c * 7u);
    float dx = (capsules[(b + 3u)] - capsules[(b + 0u)]);
    float dy = (capsules[(b + 4u)] - capsules[(b + 1u)]);
    float dz = (capsules[(b + 5u)] - capsules[(b + 2u)]);
    float len2 = (((dx * dx) + (dy * dy)) + (dz * dz));
    float t = 0.000000;
    if ((len2 > 0.000000)) {
      t = min(1.000000, max(0.000000, (((((px - capsules[(b + 0u)]) * dx) + ((py - capsules[(b + 1u)]) * dy)) + ((pz - capsules[(b + 2u)]) * dz)) / len2)));
    }
    float ex = (px - (capsules[(b + 0u)] + (t * dx)));
    float ey = (py - (capsules[(b + 1u)] + (t * dy)));
    float ez = (pz - (capsules[(b + 2u)] + (t * dz)));
    best = max(best, (capsules[(b + 6u)] - sqrt((((ex * ex) + (ey * ey)) + (ez * ez)))));
  }
  depth[i] = best;
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.CapsuleDepth
