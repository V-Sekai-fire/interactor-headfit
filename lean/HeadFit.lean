import HeadFit.SlangCodegen.Blend
import HeadFit.SlangCodegen.BlendBackward
import HeadFit.SlangCodegen.Similarity
import HeadFit.SlangCodegen.SimilarityBackwardV
import HeadFit.SlangCodegen.SimilarityBackwardPrm
import HeadFit.SlangCodegen.SurfaceResidual
import HeadFit.SlangCodegen.LandmarkResidual
import HeadFit.SlangCodegen.Prior
import HeadFit.SlangCodegen.EnergySum
import HeadFit.SlangCodegen.Resample

/-!
# `HeadFit` — the head fit's residual kernels and their gradients

RFD 2275's method as RFD 2277 places it in dress-on (`guest/headfit/`,
`headfit.elf`): ANNY's MPFB2 head, shaped by its phenotype and local
dials, placed by a similarity (6D rotation, translation, one scale), fitted
to an avatar's face by point-to-surface distance plus sparse landmarks
under a quadratic prior, with dress-on's L-BFGS-B (`guest/drape/lbfgsb.h`)
in reverse communication. `lake exe emit_headfit` writes them for
`kernels/headfit/gen.sh`.

Forward, then backward:

    hf_blend                   coeffs              -> v
    hf_similarity              prm, v              -> pos
    hf_surface_residual        pos, q, n, w        -> e_s, dpos
    hf_landmark_residual       pos, marks          -> e_k, dpos (accumulate)
    hf_prior                   x                   -> e_p, g
    hf_energy_sum              e_s | e_k | e_p     -> the loss (df32)
    hf_similarity_backward_prm dpos                -> dprm (rot6, t, s)
    hf_similarity_backward_v   dpos                -> dv
    hf_blend_backward          dv                  -> dcoeffs

and the transfer, `hf_resample`: a shape's deltas carried to the target's
vertices at their ray hits, through the fit's rotation and scale.

Not differentiated here: the phenotype anchor rule (coeffs from the six
axes, host scalars, as `Anny` leaves it) and the correspondence search
(closest points and normals, held fixed within a round).
-/
