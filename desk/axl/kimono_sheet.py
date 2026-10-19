"""Kimono through the real auto-rig: clay orbit sheet of the face mesh, rest / own eye_blink /
generated EyeClosedLeft+Right / generated smile, 5 views. Mitsuba 3."""
import sys, numpy as np, mitsuba as mi
from PIL import Image, ImageDraw
sys.argv = ["x"]; sys.path.insert(0, "/Users/ernest.lee/Desktop/stk-work/axl")
import cues, case2
mi.set_variant("scalar_rgb")
J, acc = cues.load_glb("/Users/ernest.lee/Desktop/stk-work/ue/e2e/kimono.tpose.vrm")
names = case2.names_of(J, J["meshes"][0])
P, _, tris, *_ = cues.mesh_world(J, acc, 0, 0)
T = lambda n: cues.mesh_world(J, acc, 0, names.index(n))[1]  # noqa: E731
states = {"rest": 0 * P, "own eye_blink": T("eye_blink"), "generated blink": T("EyeClosedLeft") + T("EyeClosedRight"),
          "generated smile": T("EyeClosedSmileLeft") + T("EyeClosedSmileRight")}
eyeband = P[np.linalg.norm(T("eye_blink"), axis=1) > 1e-6]
centre = 0.5 * (eyeband.min(0) + eyeband.max(0))
fwd = np.array([0, 0, 1.0])
S, rows = 300, []
for st, D in states.items():
    row = []
    for az in (-50, -25, 0, 25, 50):
        a = np.radians(az); d = np.array([np.sin(a), 0.12, np.cos(a)]); d /= np.linalg.norm(d)
        Q = P + D
        sc = {"type": "scene", "integrator": {"type": "path", "max_depth": 3},
              "sensor": {"type": "perspective", "fov": 24, "to_world": mi.ScalarTransform4f().look_at(origin=(centre + d * 0.33).tolist(), target=centre.tolist(), up=[0, 1, 0]),
                         "film": {"type": "hdrfilm", "width": S, "height": S}, "sampler": {"type": "independent", "sample_count": 32}},
              "env": {"type": "constant", "radiance": {"type": "rgb", "value": 0.7}},
              "sun": {"type": "directional", "direction": [-0.3, -0.5, -1.0], "irradiance": 1.5}}
        m = mi.Mesh("f", len(Q), len(tris))
        p = mi.traverse(m); p["vertex_positions"] = Q.astype(np.float32).ravel().tolist(); p["faces"] = tris.astype(np.uint32).ravel().tolist(); p.update()
        sc["face"] = m
        sc["bsdf"] = {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.78, 0.66, 0.6]}}}
        try:
            scene = mi.load_dict(sc)
        except Exception:
            sc.pop("bsdf"); scene = mi.load_dict(sc)
        img = np.clip(np.array(mi.render(scene, spp=32))[..., :3], 0, 1)
        row.append(Image.fromarray((np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055) * 255 + .5).astype(np.uint8)))
    rows.append((st, row))
im = Image.new("RGB", (140 + 5 * S, 30 + 4 * S), (24, 24, 28)); dr = ImageDraw.Draw(im)
dr.text((8, 8), "Kimono (no lash/iris names) via auto-rig: eyes found by humanoid eye bones", fill=(230, 230, 230))
for r, (st, row) in enumerate(rows):
    dr.text((8, 30 + r * S + S // 2), st, fill=(230, 230, 230))
    for k, t in enumerate(row):
        im.paste(t, (140 + k * S, 30 + r * S))
im.save("/Users/ernest.lee/Desktop/stk-work/axl/kimono_bones_sheet.png"); print("ok")
