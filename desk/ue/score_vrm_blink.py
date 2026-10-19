"""Desk check (not committed): score FaceBlink's EyeClosedLeft/Right in a VRM 1.0 file against the
avatar's own one-eye close carried in the same mesh, and check the blink expressions.

    python3 score_vrm_blink.py fb/girl.vrm Fcl_EYE_Close_L Fcl_EYE_Close_R

rel = rms(generated - own) / rms(own) over vertices either moves (zero shape = 1). Control: the
generated right close against the own left must score worse than the generated left.
"""
import json, struct, sys
import numpy as np
import os
GEN = os.environ.get("GEN", "EyeClosed")


def load(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    at = 20 + jl
    bl = struct.unpack_from("<I", b, at)[0]
    B = b[at + 8:at + 8 + bl]

    def view(bvi, off, count, dt, n):
        bv = J["bufferViews"][int(bvi)]
        o = int(bv.get("byteOffset", 0)) + int(off)
        return np.frombuffer(B, dt, int(count) * n, o).reshape(-1, n)

    def acc(i):
        a = J["accessors"][int(i)]
        if "bufferView" not in a:
            n = {"VEC3": 3, "SCALAR": 1}[a["type"]]
            out = np.zeros((int(a["count"]), n), np.float32)
            sp = a.get("sparse")
            if sp:
                it = {5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[int(sp["indices"]["componentType"])]
                ix = view(sp["indices"]["bufferView"], sp["indices"].get("byteOffset", 0), sp["count"], it, 1)[:, 0]
                out[ix] = view(sp["values"]["bufferView"], sp["values"].get("byteOffset", 0), sp["count"], np.float32, n)
            return out
        bv = J["bufferViews"][int(a["bufferView"])]
        o = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
        n = {"VEC3": 3, "SCALAR": 1}[a["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[int(a["componentType"])]
        st = int(bv.get("byteStride", 0))
        isz = np.dtype(dt).itemsize * n
        if st and st != isz:
            raw = np.frombuffer(B, np.uint8, int(a["count"]) * st, o).reshape(-1, st)[:, :isz]
            return np.frombuffer(raw.tobytes(), dt).reshape(-1, n)
        return np.frombuffer(B, dt, int(a["count"]) * n, o).reshape(-1, n)
    return J, acc


def rel(g, r):
    m = (np.abs(g).sum(1) > 0) | (np.abs(r).sum(1) > 0)
    return float(np.sqrt(((g - r)[m] ** 2).sum() / max((r[m] ** 2).sum(), 1e-30))), int(m.sum())


def main():
    path, own_l, own_r = sys.argv[1:4]
    J, acc = load(path)
    for mi, m in enumerate(J["meshes"]):
        names = m.get("extras", {}).get("targetNames", [])
        if GEN + "Left" not in names:
            continue
        def stack(name):
            k = names.index(name)
            seen, out = set(), []
            for p in m["primitives"]:
                a = p["attributes"]["POSITION"]
                if a in seen:
                    continue
                seen.add(a)
                out.append(acc(p["targets"][k]["POSITION"]).astype(np.float64))
            return np.concatenate(out)
        def own(suffix):
            hit = [n for n in names if n.endswith(suffix)]
            if hit:
                return stack(hit[0])
            # godot-vrm's VRM 0.x -> 1.0 step renames targets morph_<k>; k = the original index
            oJ = load(sys.argv[4])[0]
            for om in oJ["meshes"]:
                on = om.get("extras", {}).get("targetNames") or (om["primitives"][0].get("extras") or {}).get("targetNames") or []
                for k, n in enumerate(on):
                    if n.endswith(suffix) and len(om["primitives"]) == len(m["primitives"]):
                        return stack("morph_%d" % k)
            raise KeyError(suffix)
        P = np.concatenate([acc(a).astype(np.float64) for a in dict.fromkeys(p["attributes"]["POSITION"] for p in m["primitives"])])
        gl, gr = stack(GEN + "Left"), stack(GEN + "Right")
        ol, orr = own(own_l), own(own_r)
        side_l = np.abs(gl).sum(1) > 0
        print("mesh %s: %d vertices; EyeClosedLeft moves %d (x mean %+.3f m), EyeClosedRight moves %d (x mean %+.3f m)"
              % (m.get("name", mi), len(P), side_l.sum(), P[side_l, 0].mean(), (np.abs(gr).sum(1) > 0).sum(), P[np.abs(gr).sum(1) > 0, 0].mean()))
        print("own left %s moves x mean %+.3f m" % (own_l, P[np.abs(ol).sum(1) > 0, 0].mean()))
        for nm, g, r in (("left", gl, ol), ("right", gr, orr)):
            v, n = rel(g, r)
            print("generated %-5s vs own: rel %.3f over %d vertices %s" % (nm, v, n, "PASS" if v < 1 else "FAIL"))
        c, _ = rel(gr, ol)
        print("control generated right vs own left: rel %.3f %s" % (c, "caught" if c > rel(gl, ol)[0] else "NOT caught"))
        pre = J["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        print("expressions:", {k: pre[k]["morphTargetBinds"] for k in ("blink", "blinkLeft", "blinkRight")})
        print("generated left down (glTF -y) mean %.1f mm" % (-gl[side_l, 1].mean() * 1e3))


main()
