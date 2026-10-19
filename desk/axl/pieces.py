"""Labelled pieces for the eye finder: every separate piece (and the skin rim) inside a found eye
region, with shape features, and the label from the avatar's own close (answer key): moves (> 30% of
the close's largest motion on a fifth of its vertices) or stays.
  python pieces.py [role] -> out/eye_pieces_<role>.parquet"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image

AXL = pathlib.Path(__file__).parent
sys.path.insert(0, str(AXL))
role = sys.argv[1] if len(sys.argv) > 1 else "tune"
import cues  # noqa: E402
import eye_finder as ef  # noqa: E402
import render_face  # noqa: E402

A = pd.read_parquet(AXL / "out/assets.parquet").set_index("uuid")
K = pd.read_parquet(AXL / ("out/eye_key_%s.parquet" % role))
Sp = pd.read_parquet(AXL / "out/splits.parquet").set_index("uuid")
R = pd.read_parquet(AXL / "out/eye_refs.parquet")
R = R[R.side == "L"].drop_duplicates("uuid").set_index("uuid")
rows = []
for k in K[K.status == "ok"].itertuples():
    u = k.uuid
    png = AXL / "rf" / (u + ".png")
    if not (png.with_suffix(".pos.npy")).exists() or not pathlib.Path(str(png) + ".kp.json").exists():
        continue
    kp = json.load(open(str(png) + ".kp.json"))["people"]
    if not kp or min(kp[0]["points"]["left_eye"][2], kp[0]["points"]["right_eye"][2]) < ef.MIN_CONF:
        continue
    pts = kp[0]["points"]
    f = json.load(open(png.with_suffix(".json")))
    img = np.asarray(Image.open(png).convert("RGB"), np.float64) / 255
    pos = np.load(png.with_suffix(".pos.npy"))
    d = np.hypot(pts["left_eye"][0] - pts["right_eye"][0], pts["left_eye"][1] - pts["right_eye"][1])
    r = float(np.clip(ef.RADIUS * d, 6, 160))
    g = ef.region(img, pts["left_eye"][:2], r, d)
    if g is None:
        continue
    a = A.loc[u]
    path = str(AXL / ("raw/files/%s.glb" % u if a.glb_status == "original" else "glb/%s.glb" % u))
    J, acc = cues.load_glb(path)
    P, D, tris, *_ = cues.mesh_world(J, acc, int(R.mesh_ordinal[u]), int(R.target_ordinal[u]))
    mv = np.linalg.norm(D, axis=1)
    moving = mv > 0.3 * mv.max()
    px, py = render_face.to_pixel(f, P)
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < f["size"]) & (iy < f["size"])
    inreg = np.zeros(len(P), bool); inreg[inside] = g[iy[inside], ix[inside]]
    seen = np.zeros(len(P), bool)
    seen[inside] = np.linalg.norm(pos[iy[inside], ix[inside]] - P[inside], axis=1) < ef.VISIBLE * f["half"]
    comp = cues.components(len(P), tris)
    big = np.bincount(comp).argmax()
    fwd, right, up = (np.array(f[k]) for k in ("fwd", "right", "up"))
    rim = inreg & seen & (comp == big)
    if not rim.any():
        continue
    rd = np.median(P[rim] @ fwd)
    cx, cy = P[rim] @ right, P[rim] @ up
    W = max(np.ptp(cx), 1e-9); H = max(np.ptp(cy), 1e-9)
    mx, my = np.median(cx), np.median(cy)
    for cp in np.unique(comp[inreg]):
        pc_ = inreg & (comp == cp) if cp != big else rim
        x, y, z = P[pc_] @ right, P[pc_] @ up, P[pc_] @ fwd
        rows.append(dict(uuid=u, geometry_key=Sp.geometry_key[u], piece=int(cp), is_skin=bool(cp == big), n=int(pc_.sum()),
                         width=float(np.ptp(x) / W), height=float(np.ptp(y) / W), aspect=float(np.ptp(x) / max(np.ptp(y), 1e-9)),
                         dx=float((np.median(x) - mx) / W), dy=float((np.median(y) - my) / W), depth=float((np.median(z) - rd) / W),
                         top=float((y.max() - my) / W), bottom=float((y.min() - my) / W), visible=float(seen[pc_].mean()),
                         whole_piece_in_region=float(pc_.sum() / max((comp == cp).sum(), 1)) if cp != big else np.nan,
                         moves=bool(moving[pc_].mean() > 0.2)))
S = pd.DataFrame(rows)
S.to_parquet(AXL / ("out/eye_pieces_%s.parquet" % role), compression="zstd", index=False)
p = S[~S.is_skin]
print("avatars %d, pieces %d (skin rims %d); pieces that move %d" % (S.uuid.nunique(), len(p), S.is_skin.sum(), p.moves.sum()))
print(p.groupby("moves")[["n", "width", "height", "aspect", "dy", "depth", "top", "bottom", "visible"]].median().round(2).to_string())
