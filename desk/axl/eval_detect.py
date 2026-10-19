"""Grade the image eye detector against the answer key (own left close), beside the geometry cues.
  python eval_detect.py [role]  -> out/eye_detect_<role>.parquet"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

AXL = pathlib.Path(__file__).parent
role = sys.argv[1] if len(sys.argv) > 1 else "tune"
K = pd.read_parquet(AXL / ("out/eye_key_%s.parquet" % role))
C = pd.read_parquet(AXL / ("out/eye_cues_%s.parquet" % role)).drop_duplicates("uuid").set_index("uuid")
rows = []
for k in K[K.status == "ok"].itertuples():
    p = AXL / "rf" / (k.uuid + ".png.kp.json")
    row = dict(uuid=k.uuid)
    if not p.exists():
        rows.append(dict(row, det_status="no detection file")); continue
    people = json.load(open(p))["people"]
    if not people:
        rows.append(dict(row, det_status="no person")); continue
    b = people[0]["points"]
    lx, ly, lc = b["left_eye"]
    rx, ry, rc = b["right_eye"]
    rows.append(dict(row, det_status="ok", eye_conf=min(lc, rc), det_dist=float(np.hypot(lx - k.lid_px, ly - k.lid_py) / k.lid_w_px),
                     det_dist_swapped=float(np.hypot(rx - k.lid_px, ry - k.lid_py) / k.lid_w_px)))
D = pd.DataFrame(rows).merge(K, on="uuid")
D["bones_dist"] = D.uuid.map(C.bones_dist)
D["pieces_dist"] = D.uuid.map(C.pieces_dist)
D.to_parquet(AXL / ("out/eye_detect_%s.parquet" % role), compression="zstd", index=False)
n = len(D)
ok = D.det_status == "ok"
print("avatars with a key: %d; detector ran: %d" % (n, ok.sum()))
for th in (0.3, 0.5, 0.8):
    sel = ok & (D.eye_conf >= th)
    hit = sel & (D.det_dist < 0.5)
    print("eye conf >= %.1f: claimed %3d, left eye within 0.5 lid width %3d (%.0f%% of claimed), on the wrong eye %d" % (
        th, sel.sum(), hit.sum(), 100 * hit.sum() / max(sel.sum(), 1), (sel & (D.det_dist_swapped < 0.5)).sum()))
det = ok & (D.eye_conf >= 0.5) & (D.det_dist < 0.5)
geo = (D.bones_dist < 0.5) | (D.pieces_dist < 0.5)
print("geometry cues %d, detector %d, either %d, of %d" % (geo.sum(), det.sum(), (geo | det).sum(), n))
