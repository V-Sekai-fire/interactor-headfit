"""Desk probe (not committed): the closed-eye line of each eye-closing shape on three-vrm-girl and Mire.

For each shape, the moving lash/lid vertices of the left eye (over 30% of the most), their final
height across the eye, and the sag: corners minus centre (+ = corners higher, a smile-down arc,
- = corners lower, the arch of a happy closed eye). Also the upper and lower lid travel.
"""
import json, re, struct, sys
import numpy as np
sys.argv = sys.argv[:1]
exec(open("probe_vrm_blink.py").read().split("def study")[0])


def stats(name, P, D, up, hz, left_sign):
    m = np.linalg.norm(D, axis=1)
    sel = (np.sign(P[:, hz]) == left_sign) & (m > 0.3 * m[np.sign(P[:, hz]) == left_sign].max())
    p, d = P[sel], D[sel]
    final = p[:, up] + d[:, up]
    x = p[:, hz]
    c = np.polyfit(x - x.mean(), final, 2)
    half = 0.5 * (x.max() - x.min())
    sag = c[0] * half * half
    down = d[:, up] < 0
    print("  %-34s moved %4d  down %4d (mean %.1f mm)  up %4d (mean %.1f mm)  closed line sag %+.1f mm over %.0f mm width"
          % (name, sel.sum(), down.sum(), -d[down, up].mean() * 1e3 if down.any() else 0, (~down).sum(),
             d[~down, up].mean() * 1e3 if (~down).any() else 0, sag * 1e3, 2 * half * 1e3))


v = "/Users/ernest.lee/Desktop/smoke-water/3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm"
J, acc = load(v)
mesh = J["meshes"][0]
names = mesh.get("extras", {}).get("targetNames") or mesh["primitives"][0]["extras"]["targetNames"]
P = acc(mesh["primitives"][0]["attributes"]["POSITION"])
print("three-vrm-girl (VRM 0.x: up Y, left -X)")
for n in ("Fcl_EYE_Close_L", "Fcl_EYE_Joy_L", "Fcl_EYE_Close", "Fcl_EYE_Joy", "Fcl_EYE_Sorrow", "Fcl_EYE_Angry", "Fcl_EYE_Surprised", "Fcl_ALL_Joy", "Fcl_ALL_Fun"):
    k = [i for i, t in enumerate(names) if t.endswith(n)][0]
    stats(n, P, acc(mesh["primitives"][0]["targets"][k]["POSITION"]), 1, 0, -1.0)

D = "dress-on/project/fixtures/headfit"
u = open(D + "/avatar_body.usda").read()
Pm = np.array([float(x) for x in re.findall(r"-?[\d.]+(?:e-?\d+)?", re.search(r"point3f\[\] points\s*=\s*\[(.*?)\]", u, re.S).group(1))]).reshape(-1, 3)
shapes = {}; cur = None
for line in open(D + "/avatar_shapes.txt"):
    if line.startswith("shape "):
        cur = line.split()[1]; shapes[cur] = np.zeros_like(Pm)
    elif cur:
        f = line.split(); shapes[cur][int(f[0])] = [float(t) for t in f[1:4]]
print("Mire (Z up, left +X)")
for n in ("face_wink_L", "face_blink", "face_smile", "face_nagomi", "face_sad", "eye_flatten", "eye_squeeze", "eye_shrink", "eye_grow"):
    stats(n, Pm, shapes[n], 2, 0, 1.0)
