import Lake
open Lake DSL

package Anny where

-- Every dependency is pinned in a V-Sekai-fire repo or fork.
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- A sibling checkout in the manifest layout (contract-manifest-taskweft).
require Drape from "../../lbfgsb/lean"

-- The rotational-star solver; the SOMA-X twist kernel takes its swing-twist split from here.
require RotationStar from git
  "https://github.com/V-Sekai-fire/contract-rotation-star.git" @ "b70e19a627b3e6de19c4b52359406f771fae51c0" / "lean"

-- The ANNY body model's forward and backward kernels (blendshapes, joint
-- regressor, 6D forward kinematics, skinning, vertex residual), for the
-- in-guest L-BFGS-B fit. A default target, so a bare `lake build` checks
-- their native_decide pins.
@[default_target] lean_lib Anny

lean_exe emit_anny where
  root := `EmitAnny
