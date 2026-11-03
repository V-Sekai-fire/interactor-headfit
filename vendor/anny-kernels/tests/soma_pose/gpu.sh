#!/bin/sh
# Run the SOMA-X pose kernels on the GPU (Godot RenderingDevice, in a window) and gate their
# vertices against SOMA-X with the same test as the CPU path.
#
#   tests/soma_pose/gpu.sh <fixture dir>          GODOT=<godot binary> (Godot 4.x with RenderingDevice)
#
# Needs build/spv-anny from kernels/anny/gen.sh and build/soma_pose_test from build.sh. Godot runs
# windowed on purpose: the GPU path is tested as an application uses it, never --headless.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
FIX=$(cd "${1:?usage: gpu.sh <fixture dir>}" && pwd)
GODOT=${GODOT:?set GODOT to a Godot 4 binary}
PROJ="$ROOT/build/soma_gpu_project"
OUT="$ROOT/build/soma_gpu_out"
mkdir -p "$PROJ" "$OUT"
printf 'config_version=5\n[application]\nconfig/name="soma_gpu"\n' > "$PROJ/project.godot"
cp "$HERE/gpu_run.gd" "$PROJ/gpu_run.gd"
SOMA_FIXTURE="$FIX" SOMA_SPV="$ROOT/build/spv-anny" SOMA_OUT="$OUT" \
	"$GODOT" --path "$PROJ" --rendering-method forward_plus -s gpu_run.gd
"$ROOT/build/soma_pose_test/soma_pose_test" "$FIX" --gpu "$OUT"
cat "$OUT/timing.txt"
