#!/bin/sh
# Re-emit the SOMA-X kernels after a deliberate change to their Lean: clear their pins (a stale pin
# stops `lake exe emit_anny` from building), run gen.sh, pin the new text, and check every pin.
#   tools/soma/reemit.sh
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-python3}
cd "$ROOT"
for f in lean/Anny/SlangCodegen/Soma*.lean; do
	"$PY" - "$f" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = re.sub(r"-- BEGIN PIN\n.*?-- END PIN\n", '-- BEGIN PIN\ndef expected : String := ""\n-- END PIN\n', s, flags=re.S)
open(p, "w", encoding="utf-8").write(s)
EOF
done
PATH="$(dirname "$(command -v "$PY")"):$PATH" kernels/anny/gen.sh
# gen.sh's inline_prelude.py rewrites one comment character of the Slang prelude on this host
# ("host - matching" becomes an em dash). Keep the committed emits of the kernels this script did
# not touch, and the committed spelling in the soma ones, so the diff is only the kernels.
git checkout -- kernels/anny/cpp/anny_*_emit.cpp
sed -i.bak 's/on the host \xe2\x80\x94 matching/on the host - matching/' kernels/anny/cpp/soma_*_emit.cpp
rm -f kernels/anny/cpp/soma_*_emit.cpp.bak
"$PY" kernels/anny/pin.py
( cd lean && lake build )
"$PY" kernels/anny/pin.py --check
