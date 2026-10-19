"""Render every distinct-geometry tune avatar with a left eye close (cues ok) for the eye detector,
and record the answer key separately: the own close's lid centre and width, in pixels.
  python batch_render.py [role]   -> rf/<uuid>.png, rf/<uuid>.json, out/eye_key_<role>.parquet"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

AXL = pathlib.Path(__file__).parent
sys.path.insert(0, str(AXL))
sys.argv = sys.argv[:2]
import cues  # noqa: E402
import render_face  # noqa: E402

role = sys.argv[1] if len(sys.argv) > 1 else "tune"
A = pd.read_parquet(AXL / "out/assets.parquet").set_index("uuid")
C = pd.read_parquet(AXL / ("out/eye_cues_%s.parquet" % role))
C = C[C.status == "ok"].drop_duplicates("geometry_key")
R = pd.read_parquet(AXL / "out/eye_refs.parquet")
R = R[R.side == "L"].drop_duplicates("uuid").set_index("uuid")
rows = []
t0 = time.time()
for n, u in enumerate(C.uuid):
    a = A.loc[u]
    path = str(AXL / ("raw/files/%s.glb" % u if a.glb_status == "original" else "glb/%s.glb" % u))
    png = AXL / "rf" / (u + ".png")
    row = dict(uuid=u)
    try:
        f = render_face.render(path, str(png), 640, 8) if not png.exists() else __import__("json").load(open(png.with_suffix(".json")))
        if f is None:
            rows.append(dict(row, status="no frame")); continue
        J, acc = cues.load_glb(path)
        P, D, *_ = cues.mesh_world(J, acc, int(R.mesh_ordinal[u]), int(R.target_ordinal[u]))
        mv = np.linalg.norm(D, axis=1)
        lid = P[mv > 0.3 * mv.max()]
        px, py = render_face.to_pixel(f, lid)
        rows.append(dict(row, status="ok", lid_px=float(np.mean(px)), lid_py=float(np.mean(py)),
                         lid_w_px=float(max(np.ptp(px), np.ptp(py), 1.0))))
    except Exception as e:
        rows.append(dict(row, status="error %s: %s" % (type(e).__name__, str(e)[:80])))
    if n % 20 == 0:
        print("%d / %d  %.0f s" % (n, len(C), time.time() - t0), flush=True)
pd.DataFrame(rows).to_parquet(AXL / ("out/eye_key_%s.parquet" % role), compression="zstd", index=False)
print(pd.DataFrame(rows).status.value_counts().to_string())
