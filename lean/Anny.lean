import Anny.SlangCodegen.Common
import Anny.SlangCodegen.Blend
import Anny.SlangCodegen.BlendBackward
import Anny.SlangCodegen.CsrGemv3
import Anny.SlangCodegen.Fk
import Anny.SlangCodegen.FkBackward
import Anny.SlangCodegen.Lbs
import Anny.SlangCodegen.CapsuleDepth
import Anny.SlangCodegen.FkLocal
import Anny.SlangCodegen.Ground
import Anny.SlangCodegen.LbsBackwardBone
import Anny.SlangCodegen.LbsBackwardBind
import Anny.SlangCodegen.VertResidual
import Anny.SlangCodegen.SomaCommon
import Anny.SlangCodegen.SomaOrient
import Anny.SlangCodegen.SomaJump
import Anny.SlangCodegen.SomaTwist
import Anny.SlangCodegen.SomaExpand
import Anny.SlangCodegen.SomaSkin
import Anny.SlangCodegen.SomaCorrL1
import Anny.SlangCodegen.SomaCorrL2

/-!
# `Anny` — the ANNY body model's forward and backward kernels

`Anny.SlangCodegen.*` holds the differentiable half of ANNY's shape and
pose (NAVER's ANNY; the forward stages restate `Sinew.SlangCodegen.Pheno`
and `Sinew.SlangCodegen.Lbs` on the drape kernels' layout): blendshapes,
the sparse joint regressor, forward kinematics from 6D joint rotations,
skinning, and the vertex residual, each with its vector-Jacobian
product, so the in-guest L-BFGS-B (`guest/drape/lbfgsb.h`, the
`lean/Drape` kernels) can fit coeffs, rot6 and trans to a target mesh
the way `AnnyInverter` does in PyTorch. `lake exe emit_anny` writes them
for `kernels/anny/gen.sh`.

The chain, forward then backward:

    anny_blend          coeffs            -> vanny
    anny_csr_gemv3      M · vanny         -> jpos
    anny_fk             rot6, trans, jpos -> world, bone
    anny_lbs            vanny, bone       -> verts
    anny_vert_residual  verts − target    -> dverts
    anny_lbs_backward_bone  dverts        -> dbone
    anny_lbs_backward_bind  dverts        -> dvanny (overwrite)
    anny_fk_backward    dbone             -> drot6, dtrans, djpos
    anny_csr_gemv3      Mᵀ · djpos        -> dvanny (accumulate)
    anny_blend_backward dvanny            -> dcoeffs

Not differentiated: the bind rotations `rb` (`pheno_skeleton_fit`'s
Newton–Schulz / Kabsch pass is a stop-gradient here) and the phenotype →
coeffs interpolation (host scalars).

Every kernel module exports `shader`, `expected` and two `native_decide`
examples pinning `emit shader = expected` and the entry point name.
`Common` is shared statement generators, not a kernel.

The `Soma*` kernels pose a SOMA-X body (ANNY identity, pose correctives)
per frame on the GPU; `SomaCommon` is their shared generators. Dispatch
order, per frame:

    soma_orient          pose, transl         -> rabs, acc/anc (public), feat
    soma_jump × rp       public acc/anc       -> public world
    soma_twist           rabs, public world   -> chan (3 twist modes)
    soma_expand          chan, public world   -> acc/anc (target rig)
    soma_jump × rt       target acc/anc       -> target world
    soma_skin            target world         -> bone
    soma_corr_l1         feat                 -> z
    soma_corr_l2         rest, z              -> bind (rest + correctives)
    anny_lbs             bind, bone           -> verts

`tools/soma/export_soma.py` writes the static tables once per identity;
`tests/soma_pose` checks the chain against SOMA-X itself.
-/
