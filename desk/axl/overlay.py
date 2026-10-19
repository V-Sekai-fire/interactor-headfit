"""Overlay for failures: render + finder region (green), finder 'eye' vertices (blue), own lid (red)."""
import json, pathlib, sys
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
AXL = pathlib.Path("/Users/ernest.lee/Desktop/stk-work/axl")
sys.path.insert(0, str(AXL))
import cues, eye_finder as ef, render_face
E = pd.read_parquet(AXL / "out/eye_e2e_tune.parquet")
A = pd.read_parquet(AXL / "out/assets.parquet").set_index("uuid")
R = pd.read_parquet(AXL / "out/eye_refs.parquet"); R = R[R.side == "L"].drop_duplicates("uuid").set_index("uuid")
bad = E[(E.status == "scored") & (E.rel > 1.2)].sample(6, random_state=0)
tiles = []
for r in bad.itertuples():
    u = r.uuid; png = AXL / "rf" / (u + ".png"); f = json.load(open(png.with_suffix(".json")))
    a = A.loc[u]; path = str(AXL / ("raw/files/%s.glb" % u if a.glb_status == "original" else "glb/%s.glb" % u))
    roles, _ = ef.roles(path, int(R.mesh_ordinal[u]), int(R.target_ordinal[u]), png)
    J, acc = cues.load_glb(path)
    P, D, *_ = cues.mesh_world(J, acc, int(R.mesh_ordinal[u]), int(R.target_ordinal[u]))
    mv = np.linalg.norm(D, axis=1); lid = mv > 0.3 * mv.max()
    im = Image.open(png).convert("RGB"); dr = ImageDraw.Draw(im)
    px, py = render_face.to_pixel(f, P)
    for sel, col in ((lid, (255, 0, 0)), (roles == "eye", (0, 80, 255)), (roles == "iris", (0, 200, 0))):
        for x, y in zip(px[sel], py[sel]):
            dr.point((x, y), fill=col)
    kp = json.load(open(str(png) + ".kp.json"))["people"][0]["points"]
    for k in ("left_eye", "right_eye"):
        x, y = kp[k][:2]; dr.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 0), width=2)
    dr.text((5, 5), "%s rel %.2f eye %d iris %d" % (u[:8], r.rel, r.n_eye, r.n_iris), fill=(255, 0, 255))
    tiles.append(im.resize((420, 420)))
sheet = Image.new("RGB", (420 * 3, 420 * 2))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * 420, (i // 3) * 420))
sheet.save(AXL / "rf_fail_sheet.png"); print("ok")
