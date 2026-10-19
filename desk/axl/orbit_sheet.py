"""Textured orbit sheet of an avatar's face mesh with glTF alpha: MASK = cutout at alphaCutoff
(default 0.5), BLEND = cutout at 0.5 too (a lash card is mostly empty texels), OPAQUE = solid.
Rows: rest, own blink, generated blink, [own smile], generated smile. Mitsuba 3, 5 views.
  env F, OWN (comma list), OWNS (optional), OUT, TITLE, CLAY=1 for untextured (cutout kept);
  run from ue/ with the axl pixi env."""
import io
import os
import sys
import tempfile

import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw

sys.argv = ["x"]
sys.path.insert(0, "/Users/ernest.lee/Desktop/stk-work/axl")
sys.path.insert(0, "/Users/ernest.lee/Desktop/stk-work/ue")
import case2  # noqa: E402
import cues  # noqa: E402

mi.set_variant("scalar_rgb")
F, OWN, OWNS, OUT = os.environ["F"], os.environ["OWN"].split(","), os.environ.get("OWNS", ""), os.environ["OUT"]
CLAY = os.environ.get("CLAY") == "1"
J, acc = cues.load_glb(F)
b = open(F, "rb").read()
jl = int.from_bytes(b[12:16], "little")
at = 20 + jl
BIN = b[at + 8:at + 8 + int.from_bytes(b[at:at + 4], "little")]
MI = next(i for i, m in enumerate(J["meshes"]) if "EyeClosedLeft" in case2.names_of(J, m))
mesh = J["meshes"][MI]
names = case2.names_of(J, mesh)
P, _, _, *_ = cues.mesh_world(J, acc, MI, 0)
T = lambda n: cues.mesh_world(J, acc, MI, next(k for k, x in enumerate(names) if x == n or x.endswith(n)))[1]  # noqa: E731
own = sum(T(n) for n in OWN)
states = {"rest": 0 * P, "own blink": own, "generated blink": T("EyeClosedLeft") + T("EyeClosedRight")}
if OWNS:
    states["own smile"] = sum(T(n) for n in OWNS.split(","))
states["generated smile"] = T("EyeClosedSmileLeft") + T("EyeClosedSmileRight")

tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
first = {}
for pr in mesh["primitives"]:
    a = int(pr["attributes"]["POSITION"])
    if a not in first:
        first[a] = sum(int(J["accessors"][x]["count"]) for x in first)
prims = []
for k, pr in enumerate(mesh["primitives"]):
    a = int(pr["attributes"]["POSITION"])
    n = int(J["accessors"][a]["count"])
    idx = acc(pr["indices"]).astype(np.int64).reshape(-1, 3)
    uv = acc(pr["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in pr["attributes"] else None
    mat = J["materials"][pr["material"]] if "material" in pr else {}
    pbr = mat.get("pbrMetallicRoughness", {})
    col = pbr.get("baseColorFactor", [1, 1, 1, 1])
    mode = mat.get("alphaMode", "OPAQUE")
    cut = float(mat.get("alphaCutoff", 0.5))
    refl = {"type": "rgb", "value": [0.78, 0.66, 0.6] if CLAY else [float(x) for x in col[:3]]}
    opac = None
    t = pbr.get("baseColorTexture")
    if t is not None and uv is not None:
        src = J["textures"][t["index"]].get("source")
        img = J["images"][src] if src is not None else {}
        if "bufferView" in img:
            bv = J["bufferViews"][img["bufferView"]]
            A = np.asarray(Image.open(io.BytesIO(BIN[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]])).convert("RGBA"))
            if not CLAY:
                p = os.path.join(tmp, "c%d.png" % k)
                Image.fromarray(A[..., :3]).save(p)
                refl = {"type": "bitmap", "filename": p, "wrap_mode": "repeat"}
            if mode in ("MASK", "BLEND"):
                m = ((A[..., 3] / 255.0 * col[3]) >= cut).astype(np.uint8) * 255
                p = os.path.join(tmp, "a%d.png" % k)
                Image.fromarray(np.stack([m] * 3, -1)).save(p)
                opac = {"type": "bitmap", "filename": p, "wrap_mode": "repeat", "filter_type": "nearest", "raw": True}
    if os.environ.get("ROLES"):  # colour by material: role from probe_lid, a hue per material
        import probe_lid
        hues = {"skin": [0.8, 0.7, 0.6], "eye": [0.1, 0.4, 1.0], "lash": [0.0, 0.8, 0.2], "iris": [1.0, 0.8, 0.0], "brow": [0.5, 0.2, 0.0]}
        refl = {"type": "rgb", "value": hues.get(probe_lid.role_of(mat.get("name", "")), [1.0, 0.0, 0.0])}
        if "Extra" in mat.get("name", ""):
            refl = {"type": "rgb", "value": [0.0, 1.0, 1.0]}
    bsdf = {"type": "twosided", "material": {"type": "diffuse", "reflectance": refl}}
    if opac is not None:
        bsdf = {"type": "mask", "opacity": opac, "material": bsdf}
    prims.append(dict(base=first[a], n=n, idx=idx, uv=uv, bsdf=bsdf, mode=mode))


def ply(path, V, uv, idx):
    with open(path, "wb") as fh:
        h = ["ply", "format binary_little_endian 1.0", "element vertex %d" % len(V), "property float x", "property float y", "property float z"]
        if uv is not None:
            h += ["property float u", "property float v"]
        h += ["element face %d" % len(idx), "property list uchar int vertex_indices", "end_header"]
        fh.write(("\n".join(h) + "\n").encode())
        X = np.hstack([V, np.stack([uv[:, 0], 1 - uv[:, 1]], 1)]) if uv is not None else V
        fh.write(X.astype("<f4").tobytes())
        Fc = np.zeros(len(idx), dtype=[("n", "u1"), ("i", "<i4", 3)])
        Fc["n"] = 3
        Fc["i"] = idx
        fh.write(Fc.tobytes())


band = P[np.linalg.norm(own, axis=1) > 1e-6]
centre = 0.5 * (band.min(0) + band.max(0))
S, rows = 300, []
for st, D in states.items():
    Q = P + D
    shapes = {}
    for k, pm in enumerate(prims):
        p = os.path.join(tmp, "%s_%d.ply" % (st.replace(" ", "_"), k))
        ply(p, Q[pm["base"]:pm["base"] + pm["n"]], pm["uv"], pm["idx"])
        shapes["p%d" % k] = {"type": "ply", "filename": p, "face_normals": True, "bsdf": pm["bsdf"]}
    row = []
    for az in (-50, -25, 0, 25, 50):
        a = np.radians(az)
        d = np.array([np.sin(a), 0.12, np.cos(a) * float(os.environ.get("FWD", "1"))])
        d /= np.linalg.norm(d)
        sc = {"type": "scene", "integrator": {"type": "path", "max_depth": 8},
              "sensor": {"type": "perspective", "fov": 24, "to_world": mi.ScalarTransform4f().look_at(origin=(centre + d * 0.33).tolist(), target=centre.tolist(), up=[0, 1, 0]),
                         "film": {"type": "hdrfilm", "width": S, "height": S}, "sampler": {"type": "independent", "sample_count": 32}},
              "env": {"type": "constant", "radiance": {"type": "rgb", "value": [1.0, 0.0, 1.0] if os.environ.get("HOLES") else 1.0}}, **shapes}
        raw = np.array(mi.render(mi.load_dict(sc), spp=32))
        img = np.clip(raw[..., :3], 0, 1)
        row.append(Image.fromarray((np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055) * 255 + .5).astype(np.uint8)))
    rows.append((st, row))
im = Image.new("RGB", (140 + 5 * S, 30 + len(rows) * S), (24, 24, 28))
dr = ImageDraw.Draw(im)
dr.text((8, 8), os.environ.get("TITLE", F) + " (alpha cutout)", fill=(230, 230, 230))
for r, (st, row) in enumerate(rows):
    dr.text((8, 30 + r * S + S // 2), st, fill=(230, 230, 230))
    for k, t in enumerate(row):
        im.paste(t, (140 + k * S, 30 + r * S))
im.save(OUT)
print("ok", OUT, "alpha modes", [pm["mode"] for pm in prims], "cutout prims", sum(pm["bsdf"]["type"] == "mask" for pm in prims))
