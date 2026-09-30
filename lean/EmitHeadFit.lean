import LeanSlang
import HeadFit

/-!
# `emit_headfit` — write the head fit's kernels as Slang

For each `HeadFit.SlangCodegen.*` kernel writes `<outDir>/<name>.slang`;
`kernels/headfit/gen.sh` takes it from there (slangc cpp + spirv, the
embedded SPIR-V, the pins).

    lake exe emit_headfit /path/to/output/dir
-/

open LeanSlang

private def kernels : List (String × SlangShaderModule) :=
  [ ("hf_blend",                   HeadFit.SlangCodegen.Blend.shader)
  , ("hf_blend_backward",          HeadFit.SlangCodegen.BlendBackward.shader)
  , ("hf_similarity",              HeadFit.SlangCodegen.Similarity.shader)
  , ("hf_similarity_backward_v",   HeadFit.SlangCodegen.SimilarityBackwardV.shader)
  , ("hf_similarity_backward_prm", HeadFit.SlangCodegen.SimilarityBackwardPrm.shader)
  , ("hf_surface_residual",        HeadFit.SlangCodegen.SurfaceResidual.shader)
  , ("hf_landmark_residual",       HeadFit.SlangCodegen.LandmarkResidual.shader)
  , ("hf_prior",                   HeadFit.SlangCodegen.Prior.shader)
  , ("hf_energy_sum",              HeadFit.SlangCodegen.EnergySum.shader)
  , ("hf_resample",                HeadFit.SlangCodegen.Resample.shader)
  ]

def main (args : List String) : IO UInt32 := do
  let outDir := args.headD "."
  IO.FS.createDirAll outDir
  for (name, m) in kernels do
    let path := outDir ++ "/" ++ name ++ ".slang"
    IO.FS.writeFile path (LeanSlang.emit m ++ "\n")
    IO.println s!"wrote {path}"
  return 0
