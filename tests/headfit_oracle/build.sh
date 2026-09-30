#!/bin/sh
# Build and run the head-fit oracle (Gate 11): host-native, clang++ -O2 -std=c++17.
# Reads a round-0 dump from tests/headfit's runner and writes
# gates/11-headfit/oracle/{gen.log,plant_grad.log,plant_objective.log}.
#
#   build/headfit_native/hf_native dump ... out=<dir>      the dump (tests/headfit)
#   DUMP=<dir> tests/headfit_oracle/build.sh               build, check, plant twice
#   ARIA=<clone> ...                                       an existing interactor-aria-lbfgspp clone
#   OUT=<dir> ...                                          logs somewhere else
#
# LBFGSpp 0.3.0 and Eigen 3.4.90 come from V-Sekai-fire/interactor-aria-lbfgspp
# @ 10086b6b, the pin tests/lbfgsb_oracle and tests/cage_oracle use, cloned into
# .deps/ (gitignored) unless ARIA names a clone already at that commit. Neither
# library reaches any guest ELF.
#
# Exit 0 only when the real run passes and both plants fail.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DEPS="$HERE/.deps"
LBFGSPP_REV=10086b6b2022802d7942ba16842c43063b3cff91
ARIA=${ARIA:-$DEPS/interactor-aria-lbfgspp}
if [ ! -d "$ARIA/.git" ]; then
  mkdir -p "$DEPS"
  git clone -q --no-checkout https://github.com/V-Sekai-fire/interactor-aria-lbfgspp.git "$ARIA"
  git -C "$ARIA" config core.longpaths true
  git -C "$ARIA" checkout -q -f "$LBFGSPP_REV"
fi
have=$(git -C "$ARIA" rev-parse HEAD)
if [ "$have" != "$LBFGSPP_REV" ]; then
  echo "ARIA $ARIA is at $have, not $LBFGSPP_REV" >&2
  exit 2
fi
EIGEN=${EIGEN:-$ARIA/thirdparty/eigen}
CXX=${CXX:-clang++}
mkdir -p "$DEPS"
"$CXX" -O2 -std=c++17 -I"$ARIA/thirdparty/LBFGSpp/include" -I"$EIGEN" "$HERE/gen.cpp" -o "$DEPS/gen"
DUMP=${DUMP:?DUMP=<dir> written by hf_native dump}
OUT=${OUT:-$ROOT/gates/11-headfit/oracle}
mkdir -p "$OUT"
set +e
"$DEPS/gen" "$DUMP" > "$OUT/gen.log" 2>&1; real=$?
"$DEPS/gen" "$DUMP" --plant grad > "$OUT/plant_grad.log" 2>&1; pg=$?
"$DEPS/gen" "$DUMP" --plant objective > "$OUT/plant_objective.log" 2>&1; po=$?
cat "$OUT/gen.log"
echo "--- plant grad (must fail): exit $pg"; grep -E "gradcheck \(|^(PASS|FAIL):" "$OUT/plant_grad.log"
echo "--- plant objective (must fail): exit $po"; grep -E "^f0|solve: \(|^(PASS|FAIL):" "$OUT/plant_objective.log"
if [ $real -eq 0 ] && [ $pg -ne 0 ] && [ $po -ne 0 ]; then
  echo "oracle: PASS (the real run passes, both plants are caught)"; exit 0
fi
echo "oracle: FAIL (real exit $real, plant grad exit $pg, plant objective exit $po)"; exit 1
