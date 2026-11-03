#!/bin/sh
# Build fit_anny with the in-guest L-BFGS-B on the CPU path and the Slang-to-CPU phenotype host.
#   tests/anny_fit/build.sh <dir holding the baked soma_pheno.h>
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)        # core/lbfgsb
CORE=$(cd "$ROOT/.." && pwd)           # core
BAKED=${1:?the directory with the baked soma_pheno.h}
OUT=${OUT:-$HERE/build}
mkdir -p "$OUT"
c++ -O2 -std=c++20 -ffp-contract=off -Wno-unused-function -I"$BAKED" -I"$CORE" -I"$CORE/../ports" -I"$ROOT/guest/drape" -I"$CORE/slang_rt" \
  -I"$ROOT/kernels/drape/cpp" "$HERE/fit_anny.cpp" "$CORE/soma_pheno_slang.cpp" "$ROOT/guest/drape/lbfgsb.cpp" "$ROOT/guest/drape/vec_cpu.cpp" -o "$OUT/fit_anny"
echo "$OUT/fit_anny"
