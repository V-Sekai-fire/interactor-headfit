"""Run RF-DETR keypoints (COCO-17) on renders; write <png>.kp.json with the best person's points.
  python detect_eyes.py a.png [b.png ...]"""
import json
import sys

from PIL import Image
from rfdetr import RFDETRKeypointPreview

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

model = RFDETRKeypointPreview()
for p in sys.argv[1:]:
    im = Image.open(p).convert("RGB")
    d = model.predict(im, threshold=0.05)
    out = {"file": p, "people": []}
    xy = getattr(d, "xy", None)  # supervision KeyPoints: xy (n, 17, 2), confidence (n, 17) per point
    kc = getattr(d, "confidence", None)
    if xy is not None:
        for i in range(len(xy)):
            out["people"].append({"score": float(kc[i][1] + kc[i][2]) / 2,  # the eyes' confidence
                                  "points": {COCO[k]: [float(xy[i][k][0]), float(xy[i][k][1]), float(kc[i][k])] for k in range(min(17, len(xy[i])))}})
    out["people"].sort(key=lambda q: -(q["score"] or 0))
    json.dump(out, open(p + ".kp.json", "w"))
    best = out["people"][0] if out["people"] else None
    print(p, "people", len(out["people"]), "best", best and round(best["score"], 3),
          best and {k: [round(v, 1) if isinstance(v, float) else v for v in best["points"][k]] for k in ("nose", "left_eye", "right_eye")})
