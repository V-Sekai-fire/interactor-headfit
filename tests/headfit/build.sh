#!/bin/sh
# Build the head fit's native check runner (Gate 11): guest/headfit's core and
# kernels (the Lean-emitted cpp, kernels/headfit/cpp), dress-on's L-BFGS-B and
# the org's rotation fitter, host-native, -ffp-contract=off as the guest builds
# them. No Python, no Eigen.
#
#   tests/headfit/build.sh            -> build/headfit_native/hf_native
#   CXX=... CC=... tests/headfit/build.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT="$ROOT/build/headfit_native"
# The shared headers are a sibling checkout in the manifest layout.
GUEST_COMMON=${GUEST_COMMON_ROOT:-$(cd "$ROOT/../../2-contract/guest-common" 2>/dev/null && pwd)}
CXX=${CXX:-clang++}
CC=${CC:-clang}
mkdir -p "$OUT"
FLAGS="-O2 -ffp-contract=off -Wall -Wextra -Werror=absolute-value -Wno-unused-parameter"
INC="-I$ROOT/guest/headfit -I$GUEST_COMMON/guest -I$ROOT/guest/drape -I$ROOT/guest/avbd/slang-rt -I$ROOT/vendor/sinew-align -I$HERE"
"$CC" -O2 -ffp-contract=off -I"$ROOT/vendor/sinew-align" -c "$ROOT/vendor/sinew-align/sinew_align.c" -o "$OUT/sinew_align.o"
"$CXX" -std=c++17 $FLAGS $INC -c "$GUEST_COMMON/guest/common/hf_geom.cpp" -o "$OUT/hf_geom.cpp.o" &
for f in guest/headfit/hf_core.cpp guest/headfit/hf_kernels.cpp \
         guest/drape/lbfgsb.cpp guest/drape/vec_cpu.cpp tests/headfit/hf_io.cpp tests/headfit/hf_native.cpp; do
  "$CXX" -std=c++17 $FLAGS $INC -c "$ROOT/$f" -o "$OUT/$(basename "$f").o" &
done
wait
"$CXX" -o "$OUT/hf_native" "$OUT"/*.o -lm
echo "built $OUT/hf_native"
