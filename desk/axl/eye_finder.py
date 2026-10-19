"""Find an avatar's eyes without names: RF-DETR's eye keypoints on a straight-on render, grown into
the eye region by colour, projected onto the face mesh.

roles(path, mi, png) -> per-vertex roles ("skin" / "eye" / "iris") for mesh `mi` in score.load_case's
vertex order, or (None, why). Uses only the mesh, the skeleton (render frame) and the render: never
an expression. eye = face vertices visible inside the region; iris = vertices of the mesh's other
connected pieces inside it, which stay still (191 of 206 such pieces do not move in the tune
avatars' own closes)."""
import json
import pathlib
import sys

import numpy as np
from PIL import Image

AXL = pathlib.Path(__file__).parent
sys.path.insert(0, str(AXL))
import cues  # noqa: E402
import render_face  # noqa: E402

MIN_CONF = 0.5
RADIUS = 0.45  # of the distance between the eyes
SKIN_DIFF = 0.15  # RGB distance (0..1) from the cheek colour that counts as eye
VISIBLE = 0.01  # of the render's half-size: a vertex this close to the surface seen at its pixel


def region(img, p, r, d):
    from scipy import ndimage
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= r * r
    cy, cx = int(p[1] + 0.9 * r), int(p[0])
    h = max(2, int(0.15 * r))
    patch = img[max(0, cy - h):cy + h, max(0, cx - h):cx + h].reshape(-1, 3)
    if not len(patch):
        return None
    skin = np.median(patch, 0)
    mask = disc & (np.linalg.norm(img - skin, axis=2) > SKIN_DIFF)
    lab, n = ndimage.label(mask)
    inner = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= (0.25 * r) ** 2
    keep = np.unique(lab[inner & mask])
    keep = keep[keep > 0]
    if not len(keep):
        return None
    return ndimage.binary_dilation(np.isin(lab, keep), iterations=2)


def roles(path, mi, ti, png):
    png = pathlib.Path(png)
    f = json.load(open(png.with_suffix(".json")))
    kp = json.load(open(str(png) + ".kp.json"))["people"]
    if not kp:
        return None, "no person found"
    pts = kp[0]["points"]
    if min(pts["left_eye"][2], pts["right_eye"][2]) < MIN_CONF:
        return None, "eyes not found with confidence %.2f" % MIN_CONF
    img = np.asarray(Image.open(png).convert("RGB"), np.float64) / 255
    pos = np.load(png.with_suffix(".pos.npy"))
    d = np.hypot(pts["left_eye"][0] - pts["right_eye"][0], pts["left_eye"][1] - pts["right_eye"][1])
    r = float(np.clip(RADIUS * d, 6, 160))
    regs = [region(img, pts[k][:2], r, d) for k in ("left_eye", "right_eye")]
    if any(g is None for g in regs):
        return None, "no eye region around the keypoint"
    J, acc = cues.load_glb(path)
    P, D, tris, *_ = cues.mesh_world(J, acc, mi, ti)
    px, py = render_face.to_pixel(f, P)
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < f["size"]) & (iy < f["size"])
    R = np.array(["skin"] * len(P), dtype=object)
    comp = cues.components(len(P), tris)
    big = np.bincount(comp).argmax()
    seen = np.zeros(len(P), bool)
    seen[inside] = np.linalg.norm(pos[iy[inside], ix[inside]] - P[inside], axis=1) < VISIBLE * f["half"]
    for g in regs:
        inreg = np.zeros(len(P), bool)
        inreg[inside] = g[iy[inside], ix[inside]]
        rim = inreg & seen & (comp == big)
        R[rim] = "eye"
        if not rim.any():
            continue
        # Separate pieces in the region stay still (eyeball, iris, highlight): over the tune avatars'
        # own closes 191 of 206 such pieces do not move; the lid skin does the closing.
        R[inreg & (comp != big)] = "iris"
    if not (R == "eye").any() and not (R == "iris").any():
        return None, "no face vertex in the eye region"
    return R, ""
