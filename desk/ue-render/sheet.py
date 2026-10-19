"""Orbit-view contact sheets: rest, the avatar's own left blink, and the generated anime blink.

Views come from sphere_hammersley (anny-render-corpus render_view.py), framed on the head so it
fills the frame. VRMs render textured (alpha-masked materials through Mitsuba's mask BSDF);
Mire's fixture has no UVs and renders in clay.

    pixi run python sheet.py <out dir> [--views 12] [--size 320] [--spp 32]
"""
import argparse, math, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = pathlib.Path.home() / "Desktop/smoke-water/6-datasource/anny-render-corpus"
sys.path.insert(0, str(CORPUS))
sys.path.insert(0, str(HERE.parent))
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")
import vrm_load  # noqa: E402
from render_view import sphere_hammersley  # noqa: E402
import probe_lid  # noqa: E402

PARAMS = (0.3, 0.6, 0.25, 0.0)  # tuned on three-vrm-girl and AliciaSolid (probe_lid.py)
S = pathlib.Path.home() / "Desktop/smoke-water"
VRMS = {
    "three-vrm-girl": S / "3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm",
    "AliciaSolid": S / "4-entities/godot-vrm/vrm_samples/AliciaSolid_vrm-0.51.vrm",
}


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def look(eye, target, up):
    return mi.ScalarTransform4f().look_at(origin=[float(v) for v in eye], target=[float(v) for v in target],
                                         up=[float(v) for v in up])


def views(n):
    """Unit directions in a Y-up frame from the Z-up sequence: (x, y, z)_zup -> (x, z, -y)."""
    out = []
    for i in range(n):
        yaw, pitch = sphere_hammersley(i, n)
        d = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        out.append(np.array([d[0], d[2], -d[1]]))
    return out


def mesh_dict(name, pos, faces, uv, bsdf):
    m = mi.Mesh(name, vertex_count=len(pos), face_count=len(faces), has_vertex_normals=False,
                has_vertex_texcoords=uv is not None)
    p = mi.traverse(m)
    p["vertex_positions"] = mi.Float(pos.astype(np.float32).reshape(-1))
    p["faces"] = mi.UInt(faces.astype(np.uint32).reshape(-1))
    if uv is not None:
        p["vertex_texcoords"] = mi.Float(uv.astype(np.float32).reshape(-1))
    p.update()
    m.set_bsdf(mi.load_dict(bsdf))
    return m


import states as ST  # noqa: E402


def vrm_scenes(name):
    """{column: (prims, pts) or None} posed at rest with each column's delta on the face mesh."""
    path = ST.REFS[name][0]
    av = vrm_load.Avatar(str(path))
    c, st = ST.states(name)
    mesh = c["mesh_index"]
    out = {}
    base = [dict(p) for p in av.prims]
    for col in ST.COLUMNS:
        d = st[col]
        if d is None:
            out[col] = None
            continue
        av.prims = [dict(p, pos=p["pos"] + d) if p["mesh"] == mesh else dict(p) for p in base]
        out[col] = av.pose(None)[:2]
    av.prims = base
    face_prims = [k for k, p in enumerate(base) if p["mesh"] == mesh]
    return out, face_prims


def render_vrm(states, face_prims, dirs, size, spp):
    prims0, pts = states["rest"]
    head = np.concatenate([prims0[k]["pos"] for k in face_prims])
    lo, hi = head.min(0), head.max(0)
    centre = 0.5 * (lo + hi)
    rad = 0.5 * np.linalg.norm(hi - lo)
    fov = 30.0
    dist = rad / math.sin(math.radians(fov) / 2) / 0.94 * 0.9
    tex = {}
    out = {}
    for state, val in states.items():
        if val is None:
            out[state] = None
            continue
        prims = val[0]
        shapes = {}
        for j, p in enumerate(prims):
            # head-only: a primitive with nothing above the face's lowest point is the body
            if p["pos"][:, 1].max() < lo[1]:
                continue
            p = dict(p)
            keep = (p["pos"][p["faces"]][:, :, 1].max(1) >= lo[1] - 0.25 * (hi[1] - lo[1]))
            p["faces"] = p["faces"][keep]
            if len(p["faces"]) == 0:
                continue
            mat = p["mat"]
            uv = p["uv"]
            if mat["tex"] is not None and uv is not None:
                key = id(mat["tex"])
                if key not in tex:
                    t = mat["tex"].astype(np.float32) / 255.0
                    rgb = (srgb_to_linear(t[:, :, :3]) * mat["base"][None, None, :]).astype(np.float32)
                    tex[key] = (mi.Bitmap(rgb, pixel_format=mi.Bitmap.PixelFormat.RGB),
                                mi.Bitmap(np.ascontiguousarray(t[:, :, 3:4]), pixel_format=mi.Bitmap.PixelFormat.Y))
                refl = {"type": "bitmap", "bitmap": tex[key][0], "raw": True, "filter_type": "bilinear"}
                uvf = uv.copy()
                uvf[:, 1] = 1.0 - uvf[:, 1]
            else:
                refl = {"type": "rgb", "value": [float(v) for v in srgb_to_linear(np.asarray(mat["base"]))]}
                uvf = None
            bsdf = {"type": "twosided", "material": {"type": "diffuse", "reflectance": refl}}
            if mat["alpha"] != "OPAQUE" and mat["tex"] is not None and uv is not None:
                bsdf = {"type": "mask", "opacity": {"type": "bitmap", "bitmap": tex[id(mat["tex"])][1], "raw": True,
                                                    "filter_type": "bilinear"}, "material": bsdf}
            shapes[f"p{j}"] = mesh_dict(f"p{j}", p["pos"], p["faces"], uvf, bsdf)
        imgs = []
        for d in dirs:
            eye = centre + d * dist
            up = [0, 1, 0] if abs(d[1]) < 0.99 else [0, 0, 1]
            scene = mi.load_dict({"type": "scene", "integrator": {"type": "path", "max_depth": 8},
                                  "sensor": {"type": "perspective", "fov": fov, "to_world": look(eye, centre, up),
                                             "film": {"type": "hdrfilm", "width": size, "height": size},
                                             "sampler": {"type": "independent", "sample_count": spp}},
                                  "env": {"type": "constant", "radiance": {"type": "rgb", "value": 0.8}},
                                  "sun": {"type": "directional", "direction": [-0.3, -0.6, -1.0], "irradiance": 1.2},
                                  **shapes})
            img = np.clip(np.array(mi.render(scene, spp=spp, seed=0))[..., :3], 0, 1)
            imgs.append((np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8))
        out[state] = imgs
    return out


def render_mire(dirs, size, spp):
    import re
    c, st = ST.states("Mire")
    u = open(HERE.parent / "dress-on/project/fixtures/headfit/avatar_body.usda").read()
    idx = np.array([int(x) for x in re.search(r"int\[\] faceVertexIndices\s*=\s*\[(.*?)\]", u, re.S).group(1).split(",")]).reshape(-1, 3)
    toy = lambda P: np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)  # noqa: E731
    eye_v = np.where(c["roles"] == "eye")[0]
    head_v = np.where(c["P"][:, 2] > c["P"][eye_v, 2].min() - 0.06)[0]
    P = toy(c["P"])[head_v]
    lo, hi = P.min(0), P.max(0)
    centre = 0.5 * (lo + hi); rad = 0.5 * np.linalg.norm(hi - lo)
    fov = 30.0; dist = rad / math.sin(math.radians(fov) / 2) / 0.94 * 0.9
    import os
    if os.environ.get("EYEZOOM"):  # close-up of the avatar's left eye
        L = toy(c["P"])[eye_v[c["P"][eye_v, 0] > 0]]
        centre = 0.5 * (L.min(0) + L.max(0)); dist = 0.16; fov = 22.0
    tri_eye = np.isin(idx, eye_v).all(1)
    keep = np.isin(idx, head_v).all(1)
    out = {}
    for state in ST.COLUMNS:
        if st[state] is None:
            out[state] = None
            continue
        pos = toy(c["P"] + st[state])
        shapes = {"skin": mesh_dict("skin", pos, idx[~tri_eye & keep], None,
                                    {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.75, 0.62, 0.55]}}}),
                  "eye": mesh_dict("eye", pos, idx[tri_eye], None,
                                   {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.15, 0.25, 0.6]}}})}
        imgs = []
        for d in dirs:
            eye = centre + d * dist
            up = [0, 1, 0] if abs(d[1]) < 0.99 else [0, 0, 1]
            scene = mi.load_dict({"type": "scene", "integrator": {"type": "path", "max_depth": 4},
                                  "sensor": {"type": "perspective", "fov": fov, "to_world": look(eye, centre, up),
                                             "film": {"type": "hdrfilm", "width": size, "height": size},
                                             "sampler": {"type": "independent", "sample_count": spp}},
                                  "env": {"type": "constant", "radiance": {"type": "rgb", "value": 0.8}},
                                  "sun": {"type": "directional", "direction": [-0.3, -0.6, -1.0], "irradiance": 1.2},
                                  **shapes})
            img = np.clip(np.array(mi.render(scene, spp=spp, seed=0))[..., :3], 0, 1)
            imgs.append((np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8))
        out[state] = imgs
    return out


def sheet(name, rows, size, path, note, cols=6):
    """Views wrap into bands of `cols`; each band stacks the states for the same views."""
    states = list(rows)
    n = len(rows["rest"])
    bands = (n + cols - 1) // cols
    lab, gap, top = 130, 12, 34
    W = lab + cols * size
    H = top + bands * (len(states) * size + gap)
    im = Image.new("RGB", (W, H), (24, 24, 28))
    dr = ImageDraw.Draw(im)
    dr.text((8, 10), f"{name}: left eye, plain and smile close, {n} sphere_hammersley views {note}", fill=(230, 230, 230))
    for b in range(bands):
        y0 = top + b * (len(states) * size + gap)
        for r, s in enumerate(states):
            dr.text((8, y0 + r * size + size // 2), s, fill=(230, 230, 230))
            if rows[s] is None:
                dr.text((lab + 8, y0 + r * size + size // 2), "this avatar has no shape of this kind", fill=(150, 150, 150))
                continue
            for k in range(cols):
                v = b * cols + k
                if v >= n:
                    break
                im.paste(Image.fromarray(rows[s][v]), (lab + k * size, y0 + r * size))
                if r == 0:
                    dr.text((lab + k * size + 4, y0 + 2), f"view {v}", fill=(20, 20, 20))
    im.save(path)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dirs = views(a.views)
    notes = {"three-vrm-girl": "(plain + smile held out in turn)", "AliciaSolid": "(plain held out in turn; no own smile)",
             "Mire": "(smile held out in turn; no own plain; clay, the fixture has no UVs)"}
    for name in ("three-vrm-girl", "AliciaSolid"):
        if a.only and a.only != name:
            continue
        st, face = vrm_scenes(name)
        rows = render_vrm(st, face, dirs, a.size, a.spp)
        sheet(name, rows, a.size, out / f"sheet2-{name}.png", notes[name])
    if not a.only or a.only == "Mire":
        rows = render_mire(dirs, a.size, a.spp)
        sheet("Mire", rows, a.size, out / "sheet2-Mire.png", notes["Mire"])
