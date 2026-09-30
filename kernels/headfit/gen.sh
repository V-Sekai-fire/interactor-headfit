#!/usr/bin/env bash
# The head fit's kernels, from Lean to both targets (the kernels/anny/gen.sh
# shape), with no Python anywhere (RFD 2239; RFD 2277's ruling).
#
#   kernels/headfit/gen.sh               # emit from Lean, then cpp + spirv + embed
#   kernels/headfit/gen.sh --no-emit     # use the committed slang/ (no lake)
#   kernels/headfit/gen.sh --check       # emit into a temp dir; fail if slang/ or the pins differ
#
#   Lean (lean/, `lake exe emit_headfit`)     ->  slang/<k>.slang   (committed)
#     slangc -target cpp    ->  cpp/<k>_emit.cpp                    (committed; prelude inlined by awk)
#     slangc -target spirv  ->  <build>/spv-headfit/<k>.spv          (build artefact)
#       embed_spv.exs       ->  <build>/headfit_kernels.inc          (build artefact)
#     pin.exs               ->  the PIN blocks in lean/HeadFit/SlangCodegen/*.lean
#
# headfit.elf runs the cpp emits (the CPU path the Skateboard runs on); the
# SPIR-V is rule 2's second target, compiled and embedded here so a GPU path
# has its kernels, and so a kernel slangc cannot compile for the GPU fails now.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LEAN="${CLOTH_LEAN:-$ROOT/lean}"
BUILD="${BUILD_DIR:-$ROOT/build}"
SPV="$BUILD/spv-headfit"
SLANGC="${SLANGC:-slangc}"
command -v "$SLANGC" >/dev/null 2>&1 || SLANGC="$HOME/scoop/apps/vulkan/current/Bin/slangc"

MODE=emit
case "${1:-}" in
	--no-emit) MODE=none ;;
	--check) MODE=check ;;
	"") ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
esac

ALL=$(grep -v '^#' "$HERE/kernels.txt" | awk 'NF {print $1}' | tr '\n' ' ')

if [ "$MODE" = emit ] || [ "$MODE" = check ]; then
	command -v lake >/dev/null 2>&1 || { echo "error: lake not on PATH (or pass --no-emit)" >&2; exit 1; }
	FROM="$(mktemp -d)"
	echo "== emitting Slang from Lean at $LEAN =="
	( cd "$LEAN" && lake exe emit_headfit "$FROM" >/dev/null )
	if [ "$MODE" = check ]; then
		bad=0
		for k in $ALL; do
			cmp -s "$FROM/$k.slang" "$HERE/slang/$k.slang" || { echo "DIFFERS: $k"; bad=1; }
		done
		elixir "$HERE/pin.exs" --check || bad=1
		[ "$bad" = 0 ] && echo "check: slang/ and the pins match a fresh emission"
		exit $bad
	fi
	mkdir -p "$HERE/slang"
	for k in $ALL; do cp "$FROM/$k.slang" "$HERE/slang/$k.slang"; done
	echo "== $(echo $ALL | wc -w) kernels into slang/ =="
fi

# The inline prelude a reference (Windows) slangc writes, taken from a committed
# emit of another family: everything above the emit body.
REF="$ROOT/kernels/anny/cpp/anny_blend_emit.cpp"
PRELUDE="$(mktemp)"
awk '/^#ifdef SLANG_PRELUDE_NAMESPACE$/ { getline nxt; if (nxt ~ /^using namespace/) exit; print; print nxt; next } { print }' "$REF" \
	| sed '$d' > "$PRELUDE"

mkdir -p "$HERE/cpp" "$SPV"
echo "== slangc -target cpp ($(echo $ALL | wc -w)) =="
for k in $ALL; do
	# Relative paths: slangc writes the input path into #line directives.
	( cd "$HERE" && "$SLANGC" -target cpp -stage compute -entry main -o "cpp/${k}_emit.cpp" "slang/$k.slang" )
	f="$HERE/cpp/${k}_emit.cpp"
	first="$(head -n 1 "$f")"
	case "$first" in
		'#include "'*'slang-cpp-prelude.h"')
			{ cat "$PRELUDE"; tail -n +2 "$f"; } > "$f.tmp" && mv "$f.tmp" "$f" ;;
		'#ifndef SLANG_CPP_PRELUDE_H') ;;
		*) echo "unexpected first line in $f: $first" >&2; exit 1 ;;
	esac
done
rm -f "$PRELUDE"

# -fp-mode precise: no FMA contraction on the GPU, as the guest builds the cpp
# emits -ffp-contract=off (gates/5-drape/lbfgsb/README.md).
echo "== slangc -target spirv ($(echo $ALL | wc -w)) =="
rm -f "$SPV"/*.spv "$SPV"/*.refl.json
for k in $ALL; do
	"$SLANGC" -target spirv -profile sm_6_5 -stage compute -entry main -fp-mode precise \
		-reflection-json "$SPV/$k.refl.json" -o "$SPV/$k.spv" "$HERE/slang/$k.slang"
done
echo "== embedding SPIR-V =="
elixir "$HERE/embed_spv.exs" headfit_kernels "$SPV" "$BUILD/headfit_kernels.inc"

if [ "$MODE" = emit ]; then
	echo "== pins =="
	elixir "$HERE/pin.exs"
fi
