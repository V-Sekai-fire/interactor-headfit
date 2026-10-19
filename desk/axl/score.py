"""Score headfit's anime eye closes (the probe twin, probe_close4h) against every Articulation-XL
avatar that has its own eye-close shape and named eye materials on that mesh.

Frame: glTF Y up; the avatar's left is the mean (left - right) bone position over its skeleton's
left/right-named bones (eyes, arms, legs), snapped to +-X / +-Z; forward = left x up. A reference's side comes from its name (eye_refs) and
is checked against where it moves. Writes out/eye_scores.parquet (a derived claim: rule + params)."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
UE = pathlib.Path.home() / "Desktop/stk-work/ue"
sys.path.insert(0, str(UE))
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:] if a.startswith("--")]
import case2  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_close3h as p3  # noqa: E402
import probe_named as pn  # noqa: E402
import probe_lid  # noqa: E402

RULE = "named eyes (headfit dev/anime-named-eyes): lash or iris named, lower hull, iris cap on plain"
P_PLAIN = (0.4, 1.0, 0.0, 0.5)
P_SMILE = (0.5, 1.0, 0.2, 0.5)
ROLE = dict(pd.read_parquet(ROOT / "out/materials.parquet").pipe(lambda m: m.assign(k=list(zip(m.uuid, m.mesh_ordinal, m.primitive_ordinal)))).set_index("k").role)


LR = [(r"(^|[^a-z])(left|l)([^a-z]|$)|left", r"(^|[^a-z])(right|r)([^a-z]|$)|right")]
PARTS = r"eye|shoulder|clavicle|upperarm|upper_arm|arm|hand|thigh|upperleg|upper_leg|leg|foot|knee|elbow|wrist"


def node_world(J):
    """World matrix of every node (glTF TRS / matrix, parents first)."""
    import re as _re  # noqa: F401
    par = {}
    for i, n in enumerate(J.get("nodes", [])):
        for ch in n.get("children", []):
            par[ch] = i
    cache = {}

    def local(n):
        if "matrix" in n:
            return np.array(n["matrix"], float).reshape(4, 4).T
        t = np.array(n.get("translation", [0, 0, 0]), float)
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        S = np.diag(n.get("scale", [1, 1, 1]))
        M = np.eye(4); M[:3, :3] = R @ S; M[:3, 3] = t
        return M

    def world(i):
        if i not in cache:
            cache[i] = (world(par[i]) if i in par else np.eye(4)) @ local(J["nodes"][i])
        return cache[i]
    return {i: world(i) for i in range(len(J.get("nodes", [])))}


def skeleton_left(J):
    """The avatar's left, from left/right-named bones of the same part: mean (left - right)."""
    import re
    W = node_world(J)
    names = {i: n.get("name", "").lower() for i, n in enumerate(J.get("nodes", []))}
    pairs = []
    for i, a in names.items():
        if not re.search(PARTS, a):
            continue
        for side_l in ("left", "_l", ".l", "l_", " l"):
            if side_l in a:
                b = a.replace("left", "right") if side_l == "left" else a.replace(side_l, side_l.replace("l", "r"))
                j = [k for k, n in names.items() if n == b and k != i]
                if j:
                    pairs.append(W[i][:3, 3] - W[j[0]][:3, 3])
                break
    if not pairs:
        return None
    v = np.mean(pairs, 0)
    return v if np.linalg.norm(v[[0, 2]]) > 1e-9 else None


def load_case(uuid, path, mi, ti, roles=None):
    J, acc = case2._load(path)
    m = J["meshes"][mi]
    first, P, D, R, tris = {}, [], [], [], []
    for pi, pr in enumerate(m["primitives"]):
        a = int(pr["attributes"]["POSITION"])
        if a not in first:
            first[a] = sum(len(x) for x in P)
            P.append(acc(a))
            D.append(acc(pr["targets"][ti]["POSITION"]) if ti < len(pr.get("targets", [])) else np.zeros_like(P[-1]))
            R.append(np.array(["skin"] * len(P[-1]), dtype=object))
    P, D, R = np.concatenate(P), np.concatenate(D), np.concatenate(R)
    # into world space through the mesh's node (FBX exports often rotate Z-up to Y-up and scale cm)
    nodes = [i for i, n in enumerate(J.get("nodes", [])) if n.get("mesh") == mi]
    if nodes:
        M = node_world(J)[nodes[0]]
        P = P @ M[:3, :3].T + M[:3, 3]
        D = D @ M[:3, :3].T
    for pi, pr in enumerate(m["primitives"]):
        base = first[int(pr["attributes"]["POSITION"])]
        idx = (acc(pr["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in pr else np.arange(len(P)).reshape(-1, 3)) + base
        tris.append(idx)
        r = ROLE.get((uuid, mi, pi), "skin")
        if r != "skin":
            R[np.unique(idx.ravel())] = r
    tris = np.concatenate(tris)
    if roles is not None:  # per-vertex roles from the eye finder instead of material names
        R = np.asarray(roles, dtype=object)
    eye = np.isin(R, ["eye", "iris", "lash"])
    if eye.sum() < 6 or (R == "skin").sum() < 6:
        return None, "no eye or no skin vertices"
    # facing, snapped to +-X / +-Z
    ey = P[eye, 1]
    band = (R == "skin") & (np.abs(P[:, 1] - np.median(ey)) < max(np.ptp(ey), 1e-6) * 2)
    if band.sum() < 6:
        band = R == "skin"
    # the avatar's left from its skeleton's left/right bone names; forward = left x up
    lv = skeleton_left(J)
    if lv is None:
        return None, "no left/right bone names"
    ax = 0 if abs(lv[0]) > abs(lv[2]) else 2
    left = np.zeros(3); left[ax] = np.sign(lv[ax])
    fwd = np.cross(left, [0, 1.0, 0])
    hz = int(np.argmax(np.abs(left)))
    c = dict(name=uuid, P=P, ref=D, roles=R, up=1, fwd=fwd, left_sign=float(np.sign(left[hz])), hz=hz, tris=tris)
    # the eyes must sit either side of the middle line, else the frame (or the model) is not a face
    xs = P[eye, hz] - np.median(P[R == "skin", hz])
    if min((xs > 0).sum(), (xs < 0).sum()) < 3:
        return None, "eyes not on both sides of the middle"
    P0 = np.median(P[R == "skin", hz])
    c["P"] = P - np.eye(3)[hz] * P0  # centre the middle line on 0 (the rule splits sides by sign)
    return c, ""


def gen_both(c, kind, g):
    out = pn.close(dict(c), kind, *g)
    return out + pn.close(dict(c, left_sign=-c["left_sign"]), kind, *g)


def main():
    A = pd.read_parquet(ROOT / "out/assets.parquet").set_index("uuid")
    R = pd.read_parquet(ROOT / "out/eye_refs.parquet")
    T = pd.read_parquet(ROOT / "out/morph_targets.parquet").set_index(["uuid", "mesh_ordinal", "target_ordinal"]).name
    rows = []
    p3.HULL = "lo"
    for r in R.itertuples():
        a = A.loc[r.uuid]
        path = ROOT / ("raw/files/%s.glb" % r.uuid if a.glb_status == "original" else "glb/%s.glb" % r.uuid)
        base = dict(uuid=r.uuid, mesh_ordinal=r.mesh_ordinal, target_ordinal=r.target_ordinal, split=a.split, name=T[(r.uuid, r.mesh_ordinal, r.target_ordinal)], side=r.side, rule=RULE)
        try:
            c, why = load_case(r.uuid, str(path), r.mesh_ordinal, r.target_ordinal)
        except Exception as e:  # a file the probe loader cannot read is a coverage fact, not a crash
            c, why = None, "load error: %s" % type(e).__name__
        if c is None:
            rows.append(dict(base, status=why)); continue
        mv = np.linalg.norm(c["ref"], axis=1)
        if mv.max() < 1e-7:
            rows.append(dict(base, status="reference moves nothing")); continue
        moved = mv > 0.3 * mv.max()
        lshare = float((np.sign(c["P"][moved, c["hz"]]) == c["left_sign"]).mean())
        both = r.side.startswith("both")
        want = "R" if r.side.startswith("R") else "L"
        if not both and want == "R":
            c["left_sign"] = -c["left_sign"]  # score the right eye as "this side"
        c["sel"] = np.ones(len(c["P"]), bool) if both else (np.sign(c["P"][:, c["hz"]]) == c["left_sign"])
        kind = pc.kind_of(dict(c, sel=c["sel"] & (np.sign(c["P"][:, c["hz"]]) == c["left_sign"]) if both else c["sel"]))[0]
        g = P_PLAIN if kind == "plain" else P_SMILE
        okind = "smile" if kind == "plain" else "plain"
        go = P_SMILE if kind == "plain" else P_PLAIN
        try:
            if not pn.named(c):
                rows.append(dict(base, status="eye parts not named (no close made)")); continue
            f = gen_both if both else (lambda cc, k, gg: pn.close(dict(cc), k, *gg))
            gen = f(c, kind, g)
            rel = probe_lid.score(gen, c["ref"], c["sel"])[0]
            rel_other = probe_lid.score(f(c, okind, go), c["ref"], c["sel"])[0]
            wrong = np.nan
            if not both:
                wrong = probe_lid.score(pn.close(dict(c, left_sign=-c["left_sign"]), kind, *g), c["ref"], c["sel"])[0]
        except Exception as e:
            rows.append(dict(base, status="rule error: %s %s" % (type(e).__name__, str(e)[:60]))); continue
        rows.append(dict(base, status="scored", kind=kind, rel=rel, rel_other_kind=rel_other, rel_wrong_eye=wrong, side_moved_share=lshare,
                         named_side_agrees=bool(both or lshare > 0.5 if want == "L" else both or lshare < 0.5),
                         n_vertices=len(c["P"]), roles=json.dumps({k: int((c["roles"] == k).sum()) for k in sorted(set(c["roles"]))})))
    S = pd.DataFrame(rows).merge(pd.read_parquet(ROOT / "out/splits.parquet")[["uuid", "role", "geometry_key"]], on="uuid", how="left")
    S.to_parquet(ROOT / "out/eye_scores.parquet", compression="zstd", index=False)
    print(S.status.value_counts().to_string())
    s = S[S.status == "scored"]
    print("scored refs", len(s), "assets", s.uuid.nunique(), "geometry groups", s.geometry_key.nunique(), "by role", s.groupby("role").uuid.nunique().to_dict())
    print("named side agrees with motion:", s.named_side_agrees.mean().round(3))
    for (sp, k), g in s.groupby(["role", "kind"]):
        print("%-6s %-5s n %3d  rel median %.3f  share < 1.0: %.2f  beats other kind: %.2f  beats wrong eye: %.2f" % (
            sp, k, len(g), g.rel.median(), (g.rel < 1).mean(), (g.rel < g.rel_other_kind).mean(), (g.rel < g.rel_wrong_eye).dropna().mean() if g.rel_wrong_eye.notna().any() else np.nan))


if __name__ == "__main__":
    main()
