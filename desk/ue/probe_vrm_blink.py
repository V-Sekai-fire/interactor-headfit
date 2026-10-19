"""Desk probe (not committed): how anime avatars' own blink shapes close the eye.

For each VRM, find the left blink (VRM 0.x blink_l, VRM 1.0 blinkLeft, else blink), read the bound
morph targets of the bound meshes, and measure in the head frame (up, forward from the VRM version):
how far vertices move, where in the eye they sit, the share of upward vs downward motion, the
depth motion, and how the closed upper lid's line curves.
"""
import json, struct, sys
import numpy as np

CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

def load(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack_from("<I", b, off)[0]
    B = b[off + 8: off + 8 + bl]
    def acc(i):
        a = J["accessors"][i]; v = J["bufferViews"][a["bufferView"]]
        n = NC[a["type"]]; dt = CT[a["componentType"]]
        st = v.get("byteStride", 0); start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        cnt = a["count"]; isz = np.dtype(dt).itemsize * n
        if st and st != isz:
            raw = np.frombuffer(B, np.uint8, count=st * (cnt - 1) + isz, offset=start)
            out = np.stack([np.frombuffer(raw[k * st:k * st + isz].tobytes(), dt) for k in range(cnt)])
        else:
            out = np.frombuffer(B, dt, count=cnt * n, offset=start).reshape(cnt, n)
        if a.get("sparse"):
            raise SystemExit("sparse accessor not handled")
        return out.astype(np.float64) if dt == np.float32 else out
    return J, acc

def blink_binds(J):
    ex = J.get("extensions", {})
    if "VRMC_vrm" in ex:
        p = ex["VRMC_vrm"].get("expressions", {}).get("preset", {})
        e = p.get("blinkLeft") or p.get("blink")
        side = "left" if "blinkLeft" in p else "both"
        # node index -> mesh
        return "1.0", side, [(J["nodes"][m["node"]]["mesh"], m["index"], m["weight"]) for m in e.get("morphTargetBinds", [])]
    if "VRM" in ex:
        groups = ex["VRM"]["blendShapeMaster"]["blendShapeGroups"]
        g = [x for x in groups if x.get("presetName") == "blink_l"] or [x for x in groups if x.get("presetName") == "blink"]
        side = "left" if g and g[0]["presetName"] == "blink_l" else "both"
        return "0.x", side, [(bd["mesh"], bd["index"], bd["weight"] / 100.0) for bd in g[0]["binds"]] if g else []
    return None, None, []

def study(path):
    J, acc = load(path)
    ver, side, binds = blink_binds(J)
    if not binds:
        print(path.split("/")[-1], "no blink")
        return
    # head frame: VRM 0.x faces -Z, 1.0 faces +Z; Y up; avatar left = +X for 1.0, -X for 0.x
    fwd = -1.0 if ver == "0.x" else 1.0
    P, D = [], []
    for mesh, ti, w in binds:
        for prim in J["meshes"][mesh]["primitives"]:
            pos = acc(prim["attributes"]["POSITION"])
            if ti >= len(prim.get("targets", [])):
                continue
            d = acc(prim["targets"][ti]["POSITION"]) * w
            P.append(pos); D.append(d)
    P = np.concatenate(P); D = np.concatenate(D)
    m = np.linalg.norm(D, axis=1)
    mv = m > 1e-5
    if side == "both":
        left = (P[:, 0] * (1 if ver == "1.0" else -1)) > 0
        mv &= left
    p, d = P[mv], D[mv]; mm = np.linalg.norm(d, axis=1)
    # the eye: vertices moving over 20% of the most
    big = mm > 0.2 * mm.max()
    pb, db = p[big], d[big]
    y0, y1 = pb[:, 1].min(), pb[:, 1].max(); H = y1 - y0
    t = (pb[:, 1] - y0) / H  # 0 bottom of the moving eye area, 1 top
    up = db[:, 1] > 0
    def band(lo, hi):
        s = (t >= lo) & (t < hi)
        return (db[s, 1].mean() / H) if s.any() else float("nan")
    depth = (db[:, 2] * fwd)  # + = toward the viewer
    # closed line: upper part final height vs x (curvature sign)
    upper = t > 0.6
    xs = pb[upper, 0]; fy = pb[upper, 1] + db[upper, 1]
    curv = np.nan
    if upper.sum() > 6:
        c = np.polyfit(xs - xs.mean(), fy, 2)
        curv = c[0] * (0.5 * (xs.max() - xs.min())) ** 2  # sag at the corners vs centre (m)
    print("%-34s VRM %-3s %-4s moved %5d max %5.1f mm  eye area %4.1f mm tall; motion/H by height: bottom %+.2f  mid %+.2f  top %+.2f;"
          " up-share %.2f; depth mean %+.2f mm; closed upper line sag %+.1f mm (+ = corners higher, a frown arc)"
          % (path.split("/")[-1][:34], ver, side, mv.sum(), mm.max() * 1e3, H * 1e3, band(0, 0.33), band(0.33, 0.66), band(0.66, 1.01),
             up.mean(), depth.mean() * 1e3, curv * 1e3))

for f in sys.argv[1:]:
    try:
        study(f)
    except Exception as e:
        print(f.split("/")[-1], "ERROR", e)
