import Drape.SlangCodegen.Dsl

/-!
# `Anny.SlangCodegen.Ground` — the root as one rigid translation, from the feet, a floor and a seat

Per frame, with every bone's pose pinned (`world` from `anny_fk_local`, the root at rest):

- **height:** the smallest lift that keeps every point out of the floor and every body point out of
  the seat: the body rests on whatever it reaches first. Bones never stretch and the legs keep
  their pose, so folding knees bring the hips down until the seat holds them;
- **travel:** the support is the seat while it holds the body, else the lower foot (switching feet
  only when the other is lower by `hyst`). The support's lock point keeps its floor position and
  the body moves so it does not slide. A foot's lock is its FootBase, the toe point
  `anchor0`/`anchor1` (van den Heuvel et al. 2021, slide 12), the seat's is `anchor2`. On a switch
  the new support is locked where it already is (slides 42, 55), so nothing jumps.

Causal: a frame reads only its own pose and `state`, which the previous frame wrote; the seat test
uses the previous frame's root. `reset` 1 starts the root at the origin; 2 relocks where the body
is and keeps the travel so far (a clip looping back to its first frame). `kind` is 0 or 1 for a
foot's points, 2 for a body point. The seat is a box footprint: centre, unit forward `u`, half
depth along `u`, half width across it.

Bindings (set 0):

  0  ConstantBuffer<AnnyGroundParams> { uint N, reset, anchor0, anchor1, anchor2, has_seat;
                                        float floor_y, hyst, seat_y, seat_cx, seat_cz, seat_ux,
                                        seat_uz, half_d, half_w; }
  1  StructuredBuffer<float>   world      (J·12)
  2  StructuredBuffer<uint>    point_bone (N)
  3  StructuredBuffer<float>   point_off  (N·3)     bone-local offsets
  4  StructuredBuffer<uint>    kind       (N)
  5  RWStructuredBuffer<float> state      (7)       support, lock x, z, root x, y, z, last foot
  6  RWStructuredBuffer<float> out        (4 + N·3) root x, y, z, support; each point, root applied
-/

namespace Anny.SlangCodegen.Ground

open LeanSlang
open Drape.SlangCodegen.Dsl

def w (k : Nat) : E := at_ "world" (v "b" + u k)
def off (k : Nat) : E := at_ "point_off" (v "i" * u 3 + u k)
def ptx (a : E) : E := at_ "out" (u 4 + a * u 3)
def ptz (a : E) : E := at_ "out" (u 6 + a * u 3)
def isReset : E := ne (p "reset") (u 0)
def fresh : E := eq (p "reset") (u 1)

def shader : SlangShaderModule :=
  { structs := [ { name := "AnnyGroundParams",
                   fields := [fld "N" uT, fld "reset" uT, fld "anchor0" uT, fld "anchor1" uT,
                              fld "anchor2" uT, fld "has_seat" uT, fld "floor_y" fT, fld "hyst" fT,
                              fld "seat_y" fT, fld "seat_cx" fT, fld "seat_cz" fT, fld "seat_ux" fT,
                              fld "seat_uz" fT, fld "half_d" fT, fld "half_w" fT] } ]
  , globals := [ paramsCB "AnnyGroundParams", roF "world" 1, roU "point_bone" 2, roF "point_off" 3,
                 roU "kind" 4, rwF "state" 5, rwF "out" 6 ]
  , functions :=
      [ entry 1 [dtid]
          [ if_ (ne (.member (v "tid") "x") (u 0)) [ ret ]
          , let_ fT "rx0" (sel fresh (fl 0.0) (at_ "state" (u 3)))
          , let_ fT "rz0" (sel fresh (fl 0.0) (at_ "state" (u 5)))
          , let_ fT "y0" fltMax, let_ fT "y1" fltMax, let_ fT "yb" fltMax, let_ fT "ys" fltMax
          , for_ "i" (u 0) (p "N")
              [ let_ uT "b" (at_ "point_bone" (v "i") * u 12)
              , let_ fT "px" (w 0 * off 0 + w 1 * off 1 + w 2 * off 2 + w 9)
              , let_ fT "py" (w 3 * off 0 + w 4 * off 1 + w 5 * off 2 + w 10)
              , let_ fT "pz" (w 6 * off 0 + w 7 * off 1 + w 8 * off 2 + w 11)
              , setAt "out" (u 4 + v "i" * u 3) (v "px")
              , setAt "out" (u 5 + v "i" * u 3) (v "py")
              , setAt "out" (u 6 + v "i" * u 3) (v "pz")
              , if_ (eq (at_ "kind" (v "i")) (u 0)) [ setv "y0" (fmin (v "y0") (v "py")) ]
                  [ if_ (eq (at_ "kind" (v "i")) (u 1)) [ setv "y1" (fmin (v "y1") (v "py")) ]
                      [ setv "yb" (fmin (v "yb") (v "py"))
                      , let_ fT "qx" (v "px" + v "rx0" - p "seat_cx")
                      , let_ fT "qz" (v "pz" + v "rz0" - p "seat_cz")
                      , let_ fT "along" (v "qx" * p "seat_ux" + v "qz" * p "seat_uz")
                      , let_ fT "across" (v "qx" * p "seat_uz" - v "qz" * p "seat_ux")
                      , if_ (and_ (ne (p "has_seat") (u 0))
                              (and_ (le (fabs (v "along")) (p "half_d")) (le (fabs (v "across")) (p "half_w"))))
                          [ setv "ys" (fmin (v "ys") (v "py")) ] ] ] ]
          , let_ fT "dyf" (p "floor_y" - fmin (fmin (v "y0") (v "y1")) (v "yb"))
          , let_ fT "dys" (sel (lt (v "ys") fltMax) (p "seat_y" - v "ys") (p "floor_y" - fltMax))
          , let_ fT "dy" (fmax (v "dyf") (v "dys"))
          , let_ fT "prev" (at_ "state" (u 0))
          , let_ fT "foot" (sel isReset (sel (lt (v "y1") (v "y0")) (fl 1.0) (fl 0.0)) (at_ "state" (u 6)))
          , if_ (and_ (lt (v "foot") (fl 0.5)) (lt (v "y1") (v "y0" - p "hyst"))) [ setv "foot" (fl 1.0) ]
              [ if_ (and_ (gt (v "foot") (fl 0.5)) (lt (v "y0") (v "y1" - p "hyst"))) [ setv "foot" (fl 0.0) ] ]
          , let_ fT "s" (sel (gt (v "dys") (v "dyf")) (fl 2.0) (v "foot"))
          , let_ uT "a" (sel (gt (v "s") (fl 1.5)) (p "anchor2") (sel (gt (v "s") (fl 0.5)) (p "anchor1") (p "anchor0")))
          , let_ fT "ax" (ptx (v "a"))
          , let_ fT "az" (ptz (v "a"))
          , if_ (or_ isReset (ne (v "s") (v "prev")))
              [ setAt "state" (u 1) (v "ax" + v "rx0")
              , setAt "state" (u 2) (v "az" + v "rz0") ]
          , let_ fT "rootx" (at_ "state" (u 1) - v "ax")
          , let_ fT "rootz" (at_ "state" (u 2) - v "az")
          , setAt "state" (u 0) (v "s")
          , setAt "state" (u 3) (v "rootx")
          , setAt "state" (u 4) (v "dy")
          , setAt "state" (u 5) (v "rootz")
          , setAt "state" (u 6) (v "foot")
          , setAt "out" (u 0) (v "rootx")
          , setAt "out" (u 1) (v "dy")
          , setAt "out" (u 2) (v "rootz")
          , setAt "out" (u 3) (v "s")
          , for_ "i" (u 0) (p "N")
              [ setAt "out" (u 4 + v "i" * u 3) (at_ "out" (u 4 + v "i" * u 3) + v "rootx")
              , setAt "out" (u 5 + v "i" * u 3) (at_ "out" (u 5 + v "i" * u 3) + v "dy")
              , setAt "out" (u 6 + v "i" * u 3) (at_ "out" (u 6 + v "i" * u 3) + v "rootz") ] ] ] }

-- BEGIN PIN
def expected : String :=
"struct AnnyGroundParams {
  uint N;
  uint reset;
  uint anchor0;
  uint anchor1;
  uint anchor2;
  uint has_seat;
  float floor_y;
  float hyst;
  float seat_y;
  float seat_cx;
  float seat_cz;
  float seat_ux;
  float seat_uz;
  float half_d;
  float half_w;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyGroundParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> world;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> point_bone;
[[vk::binding(3, 0)]]
StructuredBuffer<float> point_off;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> kind;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> state;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> out;

[shader(\"compute\")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  float rx0 = ((params.reset == 1u) ? 0.000000 : state[3u]);
  float rz0 = ((params.reset == 1u) ? 0.000000 : state[5u]);
  float y0 = asfloat(2139095039u);
  float y1 = asfloat(2139095039u);
  float yb = asfloat(2139095039u);
  float ys = asfloat(2139095039u);
  for (uint i = 0u; i < params.N; ++i) {
    uint b = (point_bone[i] * 12u);
    float px = ((((world[(b + 0u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 1u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 2u)] * point_off[((i * 3u) + 2u)])) + world[(b + 9u)]);
    float py = ((((world[(b + 3u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 4u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 5u)] * point_off[((i * 3u) + 2u)])) + world[(b + 10u)]);
    float pz = ((((world[(b + 6u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 7u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 8u)] * point_off[((i * 3u) + 2u)])) + world[(b + 11u)]);
    out[(4u + (i * 3u))] = px;
    out[(5u + (i * 3u))] = py;
    out[(6u + (i * 3u))] = pz;
    if ((kind[i] == 0u)) {
      y0 = min(y0, py);
    } else {
      if ((kind[i] == 1u)) {
        y1 = min(y1, py);
      } else {
        yb = min(yb, py);
        float qx = ((px + rx0) - params.seat_cx);
        float qz = ((pz + rz0) - params.seat_cz);
        float along = ((qx * params.seat_ux) + (qz * params.seat_uz));
        float across = ((qx * params.seat_uz) - (qz * params.seat_ux));
        if (((params.has_seat != 0u) && ((abs(along) <= params.half_d) && (abs(across) <= params.half_w)))) {
          ys = min(ys, py);
        }
      }
    }
  }
  float dyf = (params.floor_y - min(min(y0, y1), yb));
  float dys = ((ys < asfloat(2139095039u)) ? (params.seat_y - ys) : (params.floor_y - asfloat(2139095039u)));
  float dy = max(dyf, dys);
  float prev = state[0u];
  float foot = ((params.reset != 0u) ? ((y1 < y0) ? 1.000000 : 0.000000) : state[6u]);
  if (((foot < 0.500000) && (y1 < (y0 - params.hyst)))) {
    foot = 1.000000;
  } else {
    if (((foot > 0.500000) && (y0 < (y1 - params.hyst)))) {
      foot = 0.000000;
    }
  }
  float s = ((dys > dyf) ? 2.000000 : foot);
  uint a = ((s > 1.500000) ? params.anchor2 : ((s > 0.500000) ? params.anchor1 : params.anchor0));
  float ax = out[(4u + (a * 3u))];
  float az = out[(6u + (a * 3u))];
  if (((params.reset != 0u) || (s != prev))) {
    state[1u] = (ax + rx0);
    state[2u] = (az + rz0);
  }
  float rootx = (state[1u] - ax);
  float rootz = (state[2u] - az);
  state[0u] = s;
  state[3u] = rootx;
  state[4u] = dy;
  state[5u] = rootz;
  state[6u] = foot;
  out[0u] = rootx;
  out[1u] = dy;
  out[2u] = rootz;
  out[3u] = s;
  for (uint i = 0u; i < params.N; ++i) {
    out[(4u + (i * 3u))] = (out[(4u + (i * 3u))] + rootx);
    out[(5u + (i * 3u))] = (out[(5u + (i * 3u))] + dy);
    out[(6u + (i * 3u))] = (out[(6u + (i * 3u))] + rootz);
  }
}"

example : LeanSlang.emit shader = expected := by native_decide
example : shader.entryPointName = "main" := by native_decide
-- END PIN

end Anny.SlangCodegen.Ground
