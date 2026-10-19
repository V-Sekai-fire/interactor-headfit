"""Which no-name eye cues exist on the Articulation-XL eye-close avatars (tune split only).

Answer key: the avatar's own one-eye close (L) -> its lid = vertices moving > 30% of the max.
Cues, per avatar, on the mesh holding that close, in world space:
  bones  - vertices whose largest skin weight is on a joint named like an eye (eye, eyeball, not
           eyebrow/eyelid/lash), split by side;
  pieces - connected components (by triangles) other than the largest, within the lid's head
           region, that the lid does not move (an eyeball/iris behind the lid);
  holes  - boundary loops of the largest component whose centre lies near the lid.
Each cue is judged by distance from its left-eye centre to the lid's centre, in lid widths."""
import pathlib
import re
import struct
import json
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]
import score  # noqa: E402

EYE_JOINT = re.compile(r"eye(?!brow|lid|lash|_?lid)|eyeball|目", re.I)
NOT_EYE = re.compile(r"brow|lid|lash|target|look", re.I)


def load_glb(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    at = 20 + jl
    B = b[at + 8:at + 8 + struct.unpack_from("<I", b, at)[0]]
    N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    T = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8, 5122: np.int16, 5120: np.int8}

    def acc(i):
        a = J["accessors"][int(i)]
        n, dt = N[a["type"]], T[int(a["componentType"])]
        cnt = int(a["count"])
        if "bufferView" not in a:
            out = np.zeros((cnt, n))
        else:
            bv = J["bufferViews"][int(a["bufferView"])]
            o = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
            isz = np.dtype(dt).itemsize * n
            st = int(bv.get("byteStride", 0)) or isz
            raw = np.frombuffer(B, np.uint8, (cnt - 1) * st + isz, o)
            idx = (np.arange(cnt)[:, None] * st + np.arange(isz)[None, :])
            out = np.frombuffer(raw[idx].tobytes(), dt).reshape(cnt, n).astype(np.float64)
            if a.get("normalized") and dt in (np.uint8, np.uint16):
                out /= np.iinfo(dt).max
        sp = a.get("sparse")
        if sp:
            ix = acc_view(J, B, sp["indices"], int(sp["count"]), T[int(sp["indices"]["componentType"])], 1)[:, 0].astype(int)
            out = out.copy(); out[ix] = acc_view(J, B, sp["values"], int(sp["count"]), dt, n)
        return out
    return J, acc


def acc_view(J, B, d, cnt, dt, n):
    bv = J["bufferViews"][int(d["bufferView"])]
    o = int(bv.get("byteOffset", 0)) + int(d.get("byteOffset", 0))
    return np.frombuffer(B, dt, cnt * n, o).reshape(cnt, n).astype(np.float64)


def mesh_world(J, acc, mi, ti):
    m = J["meshes"][mi]
    first, P, D, Jt, Wt, tris = {}, [], [], [], [], []
    for pr in m["primitives"]:
        a = int(pr["attributes"]["POSITION"])
        if a not in first:
            first[a] = sum(len(x) for x in P)
            P.append(acc(a))
            D.append(acc(pr["targets"][ti]["POSITION"]) if ti < len(pr.get("targets", [])) else np.zeros_like(P[-1]))
            at = pr["attributes"]
            Jt.append(acc(at["JOINTS_0"]) if "JOINTS_0" in at else np.zeros((len(P[-1]), 4)))
            Wt.append(acc(at["WEIGHTS_0"]) if "WEIGHTS_0" in at else np.zeros((len(P[-1]), 4)))
    for pr in m["primitives"]:
        base = first[int(pr["attributes"]["POSITION"])]
        n = len(P[list(first).index(int(pr["attributes"]["POSITION"]))])
        tris.append((acc(pr["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in pr else np.arange(n).reshape(-1, 3)) + base)
    P, D, Jt, Wt = map(np.concatenate, (P, D, Jt, Wt))
    nodes = [i for i, nd in enumerate(J.get("nodes", [])) if nd.get("mesh") == mi]
    skin = None
    if nodes:
        M = score.node_world(J)[nodes[0]]
        P = P @ M[:3, :3].T + M[:3, 3]
        D = D @ M[:3, :3].T
        if "skin" in J["nodes"][nodes[0]]:
            skin = J["skins"][J["nodes"][nodes[0]]["skin"]]
    names = [J["nodes"][j].get("name", "") for j in skin["joints"]] if skin else []
    return P, D, np.concatenate(tris), Jt.astype(int), Wt, names


def components(n, tris):
    par = np.arange(n)

    def f(a):
        r = a
        while par[r] != r:
            r = par[r]
        while par[a] != r:
            par[a], a = r, par[a]
        return r
    for a, b, c in tris:
        ra, rb, rc = f(a), f(b), f(c)
        par[ra] = rb; par[rc] = rb
    return np.array([f(i) for i in range(n)])


def boundary_loops(tris, keep):
    from collections import Counter, defaultdict
    e = Counter()
    for t in tris[keep[tris].all(1)]:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), k in e.items():
        if k == 1:
            adj[a].append(b); adj[b].append(a)
    seen, loops = set(), []
    for s in adj:
        if s in seen:
            continue
        loop, stack = [], [s]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v); loop.append(v); stack.extend(adj[v])
        loops.append(np.array(loop))
    return loops


def main(role="tune"):
    A = pd.read_parquet(ROOT / "out/assets.parquet").set_index("uuid")
    Sp = pd.read_parquet(ROOT / "out/splits.parquet").set_index("uuid")
    R = pd.read_parquet(ROOT / "out/eye_refs.parquet")
    R = R[R.side == "L"].drop_duplicates("uuid")
    R = R[R.uuid.map(Sp.role) == role]
    rows = []
    for r in R.itertuples():
        a = A.loc[r.uuid]
        path = ROOT / ("raw/files/%s.glb" % r.uuid if a.glb_status == "original" else "glb/%s.glb" % r.uuid)
        row = dict(uuid=r.uuid, geometry_key=Sp.geometry_key[r.uuid])
        try:
            J, acc = load_glb(str(path))
            P, D, tris, Jt, Wt, names = mesh_world(J, acc, r.mesh_ordinal, r.target_ordinal)
            lv = score.skeleton_left(J)
        except Exception as e:
            rows.append(dict(row, status="load error %s" % type(e).__name__)); continue
        mv = np.linalg.norm(D, axis=1)
        if mv.max() < 1e-9 or lv is None:
            rows.append(dict(row, status="no motion" if lv is not None else "no left/right bones")); continue
        lid = mv > 0.3 * mv.max()
        ax = 0 if abs(lv[0]) > abs(lv[2]) else 2
        sgn = np.sign(lv[ax])
        mid = np.median(P[:, ax])
        lc = P[lid].mean(0)
        W = max(np.ptp(P[lid][:, [ax]]), np.ptp(P[lid][:, 1]), 1e-9)
        row.update(status="ok", lid_left=bool(np.sign(lc[ax] - mid) == sgn), n_vertices=len(P))
        # bones
        dom = Jt[np.arange(len(P)), Wt.argmax(1)] if len(names) else np.full(len(P), -1)
        eyej = [k for k, n in enumerate(names) if EYE_JOINT.search(n) and not NOT_EYE.search(n)]
        on = np.isin(dom, eyej) & (Wt.max(1) > 0)
        side = np.sign(P[:, ax] - mid) == sgn
        row["eye_joints"] = "|".join(names[k] for k in eyej)
        row["bones_n"] = int((on & side).sum())
        row["bones_dist"] = float(np.linalg.norm(P[on & side].mean(0) - lc) / W) if (on & side).any() else np.nan
        # pieces
        comp = components(len(P), tris)
        big = np.bincount(comp).argmax()
        near = np.linalg.norm(P - lc, axis=1) < 1.5 * W
        cand = near & (comp != big) & ~lid & side
        row["pieces_n"] = int(cand.sum())
        row["pieces_dist"] = float(np.linalg.norm(P[cand].mean(0) - lc) / W) if cand.any() else np.nan
        # holes
        loops = boundary_loops(tris, comp == big)
        best = np.inf
        for L in loops:
            if len(L) >= 6:
                best = min(best, np.linalg.norm(P[L].mean(0) - lc) / W)
        row["holes_n"] = len(loops)
        row["holes_dist"] = float(best) if np.isfinite(best) else np.nan
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_parquet(ROOT / ("out/eye_cues_%s.parquet" % role), compression="zstd", index=False)
    ok = S[S.status == "ok"].drop_duplicates("geometry_key")
    print(S.status.value_counts().to_string())
    print("distinct-geometry avatars with a left close:", len(ok), "lid on the named left:", ok.lid_left.mean().round(3))
    for cue in ("bones", "pieces", "holes"):
        d = ok[cue + "_dist"]
        print("%-6s present %3d / %d   centre within 0.5 lid width %3d   within 1.0 %3d" % (cue, d.notna().sum(), len(ok), (d < 0.5).sum(), (d < 1.0).sum()))
    print("any cue within 0.5:", ((ok.bones_dist < 0.5) | (ok.pieces_dist < 0.5) | (ok.holes_dist < 0.5)).sum(), "of", len(ok))


if __name__ == "__main__":
    main()
