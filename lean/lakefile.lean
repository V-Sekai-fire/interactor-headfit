import Lake
open Lake DSL

package HeadFit where

-- Every dependency is pinned in a V-Sekai-fire repo or fork.
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- The solvers come from sinew-mocap/solve, which owns them (git subtree, vendor/solve); the upstreams moved to V-Sekai-archive.
require Drape from "../vendor/solve/core/lbfgsb/lean"
require RotationStar from "../vendor/solve/core/rotation-star/lean"

-- contract-anny-kernels, a git subtree (vendor/anny-kernels). Its own requires of Drape and RotationStar resolve to the two above.
require Anny from "../vendor/anny-kernels/lean"


-- The head fit's kernels (RFD66606-2275/2277, headfit.elf): the ANNY head's
-- blend and similarity, the point-to-surface and landmark residuals, the
-- prior, the df32 loss sum, their gradients, and the transfer's resample.
-- A default target, so a bare `lake build` checks their native_decide pins.
@[default_target] lean_lib HeadFit

lean_exe emit_headfit where
  root := `EmitHeadFit
