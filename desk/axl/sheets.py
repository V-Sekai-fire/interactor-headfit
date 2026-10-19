"""Contact sheets of the tune renders (face crop around the detected eyes), numbered, for labelling
anime vs realistic by eye. -> rf_sheets/sheet_<k>.png + rf_sheets/index.csv (tile number -> uuid)."""
import json, pathlib
import pandas as pd
from PIL import Image, ImageDraw
AXL = pathlib.Path(__file__).parent
K = pd.read_parquet(AXL / "out/eye_key_tune.parquet")
out = AXL / "rf_sheets"; out.mkdir(exist_ok=True)
rows, tiles = [], []
for u in K[K.status == "ok"].uuid:
    png = AXL / "rf" / (u + ".png")
    im = Image.open(png).convert("RGB")
    kp = json.load(open(str(png) + ".kp.json"))["people"]
    if kp:
        p = kp[0]["points"]; cx = (p["left_eye"][0] + p["right_eye"][0]) / 2; cy = (p["left_eye"][1] + p["right_eye"][1]) / 2
        h = max(60, 1.6 * abs(p["left_eye"][0] - p["right_eye"][0]))
        if min(p["left_eye"][2], p["right_eye"][2]) < 0.5:
            cx, cy, h = 320, 300, 200
    else:
        cx, cy, h = 320, 300, 200
    t = im.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h))).resize((200, 200))
    ImageDraw.Draw(t).text((3, 3), str(len(rows)), fill=(255, 0, 255))
    rows.append(dict(tile=len(rows), uuid=u)); tiles.append(t)
for s in range(0, len(tiles), 48):
    sh = Image.new("RGB", (200 * 8, 200 * 6), (40, 40, 40))
    for i, t in enumerate(tiles[s:s + 48]):
        sh.paste(t, ((i % 8) * 200, (i // 8) * 200))
    sh.save(out / ("sheet_%d.png" % (s // 48)))
pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
print(len(rows), "tiles")
