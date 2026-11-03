#!/bin/sh
# Build and run the SOMA-X pose parity test on the kernels' C++ emission (the CPU path).
#
#   tests/soma_pose/build.sh <fixture dir>
#
# The fixture comes from tools/soma/export_soma.py (SOMA-X on the CPU, once). Each kernel's
# kernels/anny/cpp/<k>_emit.cpp is compiled in its own unit (kernel_tu.cpp, -DSOMA_KERNEL=<n>) with
# only its *_run symbol kept global, so the emits' shared prelude names do not collide. Units build
# with -ffp-contract=off, as the guest builds the emits (no fused multiply-add the GPU path lacks).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
FIX=${1:?usage: build.sh <fixture dir>}
OUT="$ROOT/build/soma_pose_test"
CXX=${CXX:-c++}
mkdir -p "$OUT"
n=0
OBJS=""
for k in soma_orient soma_jump soma_twist soma_expand soma_skin soma_corr_l1 soma_corr_l2 anny_lbs; do
	n=$((n + 1))
	"$CXX" -O2 -std=c++17 -ffp-contract=off -Wno-everything -DSOMA_KERNEL=$n -I"$ROOT/kernels/anny/cpp" \
		-c "$HERE/kernel_tu.cpp" -o "$OUT/$k.full.o"
	if [ "$(uname)" = Darwin ]; then
		ld -r -exported_symbol "_${k}_run" "$OUT/$k.full.o" -o "$OUT/$k.o"
	else
		objcopy --keep-global-symbol="${k}_run" "$OUT/$k.full.o" "$OUT/$k.o"
	fi
	OBJS="$OBJS $OUT/$k.o"
done
"$CXX" -O2 -std=c++17 -ffp-contract=off "$HERE/test.cpp" $OBJS -o "$OUT/soma_pose_test"
"$OUT/soma_pose_test" "$FIX"
