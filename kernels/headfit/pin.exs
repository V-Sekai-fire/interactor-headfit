# Pin each head-fit kernel's emitted Slang into its Lean module (the
# kernels/anny/pin.py pattern, in Elixir: no Python, RFD 2239).
#
# Every lean/HeadFit/SlangCodegen/<Module>.lean carries a block
#
#     -- BEGIN PIN
#     def expected : String := "..."
#     example : LeanSlang.emit shader = expected := by native_decide
#     example : shader.entryPointName = "main" := by native_decide
#     -- END PIN
#
# This rewrites that block from kernels/headfit/slang/<kernel>.slang, the text
# `lake exe emit_headfit` wrote. Run it only after a deliberate change to a
# kernel (then gen.sh, then `lake build HeadFit` re-checks every pin).
#
#     elixir kernels/headfit/pin.exs           # rewrite the pins
#     elixir kernels/headfit/pin.exs --check   # exit 1 if any pin is stale
#
# SPDX-License-Identifier: Apache-2.0 OR MIT

here = Path.dirname(Path.expand(__ENV__.file))
lean = Path.join([here, "..", "..", "lean", "HeadFit", "SlangCodegen"]) |> Path.expand()

modules = [
  {"Blend", "hf_blend"},
  {"BlendBackward", "hf_blend_backward"},
  {"Similarity", "hf_similarity"},
  {"SimilarityBackwardV", "hf_similarity_backward_v"},
  {"SimilarityBackwardPrm", "hf_similarity_backward_prm"},
  {"SurfaceResidual", "hf_surface_residual"},
  {"LandmarkResidual", "hf_landmark_residual"},
  {"Prior", "hf_prior"},
  {"EnergySum", "hf_energy_sum"},
  {"Resample", "hf_resample"}
]

lean_string = fn text ->
  "\"" <> (text |> String.replace("\\", "\\\\") |> String.replace("\"", "\\\"")) <> "\""
end

pin_block = fn slang ->
  text = String.trim_trailing(slang, "\n")

  "-- BEGIN PIN\n" <>
    "def expected : String :=\n" <>
    lean_string.(text) <>
    "\n\n" <>
    "example : LeanSlang.emit shader = expected := by native_decide\n" <>
    "example : shader.entryPointName = \"main\" := by native_decide\n" <>
    "-- END PIN\n"
end

check = "--check" in System.argv()
block = ~r/-- BEGIN PIN\n.*?-- END PIN\n/s

stale =
  Enum.flat_map(modules, fn {mod, kernel} ->
    path = Path.join(lean, mod <> ".lean")
    src = File.read!(path)
    slang = File.read!(Path.join([here, "slang", kernel <> ".slang"]))
    want = pin_block.(slang)

    unless Regex.match?(block, src) do
      IO.puts(:stderr, "#{path}: no PIN block")
      System.halt(2)
    end

    new = Regex.replace(block, src, fn _ -> want end, global: false)

    cond do
      new == src ->
        []

      check ->
        IO.puts("stale: #{mod}")
        [mod]

      true ->
        File.write!(path, new)
        IO.puts("pinned: #{mod}")
        [mod]
    end
  end)

if check and stale != [] do
  System.halt(1)
end

IO.puts("pin: #{length(modules)} modules, #{length(stale)} #{if check, do: "stale", else: "rewritten"}")
