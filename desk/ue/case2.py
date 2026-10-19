"""Desk probe (not committed): one loader for any VRM 0.x / 1.0 eye-close reference, in the shape
probe_close2.close() takes. The mesh is the one holding a target whose name ends with `target`;
its primitives' vertex buffers are joined (shared buffers once), roles come from material names
(probe_lid.role_of), the frame from the VRM version (0.x faces -Z with left -X; 1.0 faces +Z with
left +X), and the triangles for eye_roles' part split."""
import json
import struct

import numpy as np

import probe_lid


def _load(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    at = 20 + jl
    B = b[at + 8:at + 8 + struct.unpack_from("<I", b, at)[0]]

    def view(bvi, off, count, dt, n, stride=0):
        bv = J["bufferViews"][int(bvi)]
        o = int(bv.get("byteOffset", 0)) + int(off)
        isz = np.dtype(dt).itemsize * n
        st = int(bv.get("byteStride", 0)) or stride
        if st and st != isz:
            raw = np.frombuffer(B, np.uint8, int(count) * st, o).reshape(-1, st)[:, :isz]
            return np.frombuffer(raw.tobytes(), dt).reshape(-1, n)
        return np.frombuffer(B, dt, int(count) * n, o).reshape(-1, n)

    def acc(i):
        a = J["accessors"][int(i)]
        n = {"VEC3": 3, "SCALAR": 1}[a["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[int(a["componentType"])]
        out = (view(a["bufferView"], a.get("byteOffset", 0), a["count"], dt, n).astype(np.float64)
               if "bufferView" in a else np.zeros((int(a["count"]), n)))
        sp = a.get("sparse")
        if sp:
            it = {5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[int(sp["indices"]["componentType"])]
            ix = view(sp["indices"]["bufferView"], sp["indices"].get("byteOffset", 0), sp["count"], it, 1)[:, 0]
            out = out.copy()
            out[ix] = view(sp["values"]["bufferView"], sp["values"].get("byteOffset", 0), sp["count"], np.float32, n)
        return out
    return J, acc


def names_of(J, m):
    return (m.get("extras") or {}).get("targetNames") or (m["primitives"][0].get("extras") or {}).get("targetNames") or []


def case(avatar, path, target):
    J, acc = _load(path)
    ver = "1.0" if "VRMC_vrm" in J.get("extensions", {}) else "0.x"
    for m in J["meshes"]:
        names = names_of(J, m)
        hit = [k for k, n in enumerate(names) if n.endswith(target)]
        if hit:
            break
    else:
        raise KeyError(target)
    k = hit[0]
    first, P, D, tris, roles = {}, [], [], [], []
    for pr in m["primitives"]:
        a = int(pr["attributes"]["POSITION"])
        if a not in first:
            first[a] = sum(len(p) for p in P)
            P.append(acc(a))
            D.append(acc(pr["targets"][k]["POSITION"]))
            roles.append(np.array(["skin"] * len(P[-1]), dtype=object))
    P, D = np.concatenate(P), np.concatenate(D)
    R = np.concatenate(roles)
    for pr in m["primitives"]:
        base = first[int(pr["attributes"]["POSITION"])]
        idx = acc(pr["indices"]).astype(np.int64).reshape(-1, 3) + base
        tris.append(idx)
        mat = J["materials"][int(pr["material"])].get("name", "") if "material" in pr else ""
        r = probe_lid.role_of(mat)
        used = np.unique(idx.ravel())
        if r != "skin":
            R[used] = r
    fwd = np.array([0, 0, -1.0]) if ver == "0.x" else np.array([0, 0, 1.0])
    c = dict(name=avatar, P=P, ref=D, roles=R, up=1, fwd=fwd, left_sign=-1.0 if ver == "0.x" else 1.0,
             tris=np.concatenate(tris), target=target, hz=0)
    c["sel"] = np.sign(P[:, 0]) == c["left_sign"]
    return c
