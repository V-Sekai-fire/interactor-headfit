import Drape.SlangCodegen.Dsl
import Anny.SlangCodegen.Common
import RotationStar.Slang.Split

/-!
# `Anny.SlangCodegen.SomaCommon` — shared pieces of the SOMA-X pose kernels

Not a kernel. Statement generators the `Soma*` kernels inline. Matrices are
nine scalar locals as in `Common` (row-major, `<pfx>(3i+j)`); a stored 3×3
is 9 floats row-major, a stored affine is 12 floats `[R | t]` (row r at
`4r … 4r+3`).

Rotations stay 3×3 matrices everywhere: no quaternion is formed in any
kernel. The twist angles SOMA-X takes through quaternions are read off the
matrix instead (`twistAngle`).
-/

namespace Anny.SlangCodegen.SomaCommon

open LeanSlang
open Drape.SlangCodegen.Dsl
open Anny.SlangCodegen.Common

/-- The 3×3 at `buf[base … base+8]` (row-major) into `pfx`. -/
def load9 (pfx buf : String) (base : E) : List St :=
  mat pfx fun i j => at_ buf (base + u (3 * i + j))

/-- Write the 3×3 `r` to `buf[base … base+8]`. -/
def store9 (buf : String) (base : E) (r : String) : List St :=
  r9.map fun k => setAt buf (base + u k) (s r k)

/-- The numerator of the twist about axis `ax` (a runtime uint, 0 = x):
    `R21 − R12`, `R02 − R20` or `R10 − R01`, which is `4 w q_ax` for the
    rotation's quaternion `(q, w)`. -/
def twistNum (r : String) (ax : E) : E :=
  sel (eq ax (u 0)) (m r 2 1 - m r 1 2)
    (sel (eq ax (u 1)) (m r 0 2 - m r 2 0) (m r 1 0 - m r 0 1))

/-- `out = ` SOMA-X's swing-twist angle of the rotation `r` about `ax`,
    `quaternion_twist_angle_xyzw(matrix_to_quaternion_xyzw(r), ax)`, from the
    matrix alone.

    SOMA-X takes the unit quaternion `(q, w)` with `w ≥ 0`, its half-angle
    root `[q, w + 1]`, and returns `4·atan2(q_ax, w + 1)`. With
    `w = ½√(1 + tr)` and `q_ax = num / 4w` (`twistNum`), multiplying both
    arguments of the atan2 by `4w > 0` gives

        4 · atan2(num, (1 + tr) + 2√(1 + tr)),

    the same angle with no quaternion. `1 + tr` is clamped at 0 as SOMA-X
    clamps it. This is not `2·atan2(q_ax, w)`: the two agree only when the
    rotation is a pure twist (the parity test measures how far apart they
    are on the clips). -/
def twistAngle (out r : String) (ax : E) : List St :=
  [ let_ fT (out ++ "_t") (fmax (fl 1.0 + m r 0 0 + m r 1 1 + m r 2 2) (fl 0.0))
  , let_ fT out (fl 4.0 * call "atan2"
      [ twistNum r ax, v (out ++ "_t") + fl 2.0 * call "sqrt" [v (out ++ "_t")] ]) ]

/-- SOMA-X's local Euler channel `ax` of the rotation `r`
    (`_local_euler_xyz_from_matrix`): channel 0 is its local-X override
    `atan2(R21, R11)`; channels 1 and 2 are `matrix_to_euler_xyz`'s
    `asin(−clamp(R20, −1, 1))` and `atan2(R10, R00)`. SOMA-X has no
    separate gimbal-lock branch, so neither has this. -/
def eulerChannel (r : String) (ax : E) : E :=
  sel (eq ax (u 0)) (call "atan2" [m r 2 1, m r 1 1])
    (sel (eq ax (u 1)) (call "asin" [- call "clamp" [m r 2 0, fl (-1.0), fl 1.0]])
      (call "atan2" [m r 1 0, m r 0 0]))

/-- Component `i` of the unit axis `ax` (a runtime uint). -/
def axisUnit (ax : E) (i : Nat) : E := sel (eq ax (u i)) (fl 1.0) (fl 0.0)

/-- `out = ` the twist angle of `r` about `ax` by the plain swing-twist split: the swing is the
    canonical rank-1 Kabsch completion (the shortest arc from the axis to its image), the twist is
    the swing's transpose times `r`, every step a 3-vector or 3×3. contract-rotation-star owns it
    (`RotationStar.Slang.Split.swingTwistAngle`, with its spec `RotationStar.SwingTwist`); this
    name keeps the kernels' call sites, and their pins check the emitted text is unchanged.

    This is the standard twist, `2·atan2(q_ax, w)`. SOMA-X's half-angle form (`twistAngle`) is
    twice the standard twist of `√r`; the two differ once `r` has swing, so the kernels select it
    with `twistForm`; the Kabsch split is the default. -/
def swingTwistAngle (out r : String) (ax : E) : List St :=
  RotationStar.Slang.Split.swingTwistAngle out r ax

end Anny.SlangCodegen.SomaCommon
