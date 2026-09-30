import Lake
open Lake DSL

package HeadFit where

-- Every dependency is pinned in a V-Sekai-fire repo or fork.
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- A sibling checkout in the manifest layout (contract-manifest-taskweft).
require Drape from "../../../2-contract/lbfgsb/lean"

-- A sibling checkout in the manifest layout (contract-manifest-taskweft).
require Anny from "../../../2-contract/anny-kernels/lean"

-- The head fit's kernels (RFD 2275/2277, headfit.elf): the ANNY head's
-- blend and similarity, the point-to-surface and landmark residuals, the
-- prior, the df32 loss sum, their gradients, and the transfer's resample.
-- A default target, so a bare `lake build` checks their native_decide pins.
@[default_target] lean_lib HeadFit

lean_exe emit_headfit where
  root := `EmitHeadFit
