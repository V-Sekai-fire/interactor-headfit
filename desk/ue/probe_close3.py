"""Desk probe (not committed): smile close v3 (and plain re-checked), leave-one-avatar-out over every
avatar with named or splittable eye parts.

Changes from probe_close2:
- eye parts: a connected eye part other than the largest is a brow when wholly above it, else an
  iris/eyeball/highlight, and stays still (Mire's 97-vertex eyeball never moves in its own closes;
  v2 slid it and the eye tore);
- outline: the eye white and iris together (Mire's lid part reaches far past her eyeball);
- smile line: c(x) = lo + k (hi - lo) + arch * H * (1 - (2u - 1)^2), u across the eye: arch > 0
  lifts the middle (Mire's ^ curve), 0 is v2's line.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
import case2  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_lid  # noqa: E402

S = "/Users/ernest.lee/Desktop/smoke-water"
TWIST = S + "/3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/VRM1_Constraint_Twist_Sample/VRM1_Constraint_Twist_Sample.vrm"
KIMONO = "/Volumes/FIRE202610/4130733195969860779.vrm"
VRM_REFS = [("three-vrm-girl", pc.GIRL, "Fcl_EYE_Close_L"), ("three-vrm-girl", pc.GIRL, "Fcl_EYE_Joy_L"),
            ("Twist", TWIST, "Fcl_EYE_Close_L"), ("Twist", TWIST, "Fcl_EYE_Joy_L"),
            ("AliciaSolid", pc.ALICIA, "eye_winkL2"), ("AliciaSolid", pc.ALICIA, "eye_winkL"),
            ("AliciaSolid", pc.ALICIA, "eye_smile"), ("AliciaSolid", pc.ALICIA, "eye_blink"),
            ("Kimono", KIMONO, "eye_blink")]
MIRE_REFS = ["face_wink_L", "face_smile", "face_blink"]


def eye_roles(c):
    if "roles2" in c:
        return c["roles2"]
    roles = c["roles"].copy()
    P, up = c["P"], c["up"]
    tris = c.get("tris")
    if tris is None or (roles == "lash").any() or (roles == "iris").any():
        c["roles2"] = roles
        return roles
    eye = roles == "eye"
    par = np.arange(len(P))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for t in tris[eye[tris].all(1)]:
        par[f(t[0])] = f(t[1])
        par[f(t[2])] = f(t[1])
    for side in (-1.0, 1.0):
        groups = {}
        for v in np.where(eye & (np.sign(P[:, c["hz"]]) == side))[0]:
            groups.setdefault(f(v), []).append(v)
        if not groups:
            continue
        big = max(groups.values(), key=len)
        btop = P[big, up].max()
        for g in groups.values():
            if g is not big:
                roles[g] = "brow" if P[g, up].min() > btop else "iris"
    c["roles2"] = roles
    return roles


def close(c, kind, k, above, below, margin, arch=0.0):
    P, up, hz = c["P"], c["up"], c["hz"]
    roles = eye_roles(c)
    side = c["left_sign"]
    da = int(np.argmax(np.abs(c["fwd"])))
    E = P[np.isin(roles, ["eye", "iris"]) & (np.sign(P[:, hz]) == side)]  # the white and iris together
    x0, x1 = E[:, hz].min(), E[:, hz].max()
    W = x1 - x0
    bins = np.linspace(x0, x1, 11)
    xs, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean()); lo.append(s[:, up].min()); hi.append(s[:, up].max())
    xs, lo, hi = map(np.array, (xs, lo, hi))
    H = float((hi - lo).max())
    front = (E[:, da] * c["fwd"][da]).min() - 0.25 * H
    out = np.zeros_like(P)
    vs = np.where((np.sign(P[:, hz]) == side) & ~np.isin(roles, ["iris", "brow"]) & (P[:, da] * c["fwd"][da] >= front))[0]
    x, h = P[vs, hz], P[vs, up]
    ox = np.maximum.reduce([np.zeros_like(x), x0 - x, x - x1]) / max(W, 1e-9)
    keep = ox <= margin
    wx = np.where(ox <= 0, 1.0, (1.0 - ox / max(margin, 1e-9)) ** 2)
    l = np.interp(x, xs, lo); t = np.interp(x, xs, hi)
    u = np.clip((x - x0) / max(W, 1e-9), 0, 1)
    cl = np.minimum(l + k * (t - l) + arch * H * (1 - (2 * u - 1) ** 2), t)
    dh = np.zeros_like(x)
    a1 = (cl <= h) & (h <= t)
    dh[a1] = -(h[a1] - cl[a1])
    a2 = (t < h) & (h <= t + above * H)
    dh[a2] = -(t[a2] - cl[a2]) * (1.0 - (h[a2] - t[a2]) / (above * H)) ** 2
    if kind == "smile":
        a3 = (l <= h) & (h < cl)
        dh[a3] = cl[a3] - h[a3]
        a4 = (l - below * H <= h) & (h < l)
        dh[a4] = (cl[a4] - l[a4]) * (1.0 - (l[a4] - h[a4]) / (below * H)) ** 2
    out[vs, up] = np.where(keep, dh * wx, 0.0)
    return out


GRID = {"plain": [g + (0.0,) for g in itertools.product((0.0, 0.1, 0.2, 0.3), (0.3, 0.6, 1.0), (0.0,), (0.1, 0.25, 0.5))],
        "smile": list(itertools.product((0.2, 0.3, 0.4, 0.5, 0.6), (0.3, 0.6, 1.0), (0.2, 0.4, 0.8), (0.1, 0.25, 0.5), (0.0, 0.1, 0.2, 0.3)))}


def rel(c, kind, g):
    return probe_lid.score(close(c, kind, *g), c["ref"], c["sel"])


def main():
    cases = {}
    for av, path, t in VRM_REFS:
        if not os.path.exists(path):
            print("missing", path)
            continue
        cases[(av, t)] = case2.case(av, path, t)
    for t in MIRE_REFS:
        cases[("Mire", t)] = pc.case("Mire", None, t)
    for key, c in list(cases.items()):
        c["kind"] = pc.kind_of(c)[0]
        same = [o for o in cases if o != key and o[0] == key[0] and o in cases and np.array_equal(cases[o]["ref"], c["ref"])]
        if same and same[0] < key:
            print("  %s %s = %s; left out" % (key[0], key[1], same[0][1]))
            del cases[key]
    out = {}
    for kind in ("plain", "smile"):
        refs = [k for k, c in cases.items() if c["kind"] == kind]
        avatars = sorted({a for a, _ in refs})
        print("\n== %s: %s" % (kind, refs))
        for hold in avatars:
            tune = [r for r in refs if r[0] != hold]
            best = min(GRID[kind], key=lambda g: max(rel(cases[r], kind, g)[0] for r in tune))
            for r in [r for r in refs if r[0] == hold]:
                v, e, m = rel(cases[r], kind, best)
                print("  HELD OUT %-15s %-16s rel %.3f (%.2f of %.2f mm)  params %s" % (r[0], r[1], v, e, m, best))
        best = min(GRID[kind], key=lambda g: max(rel(cases[r], kind, g)[0] for r in refs))
        print("  all: params %s" % (best,))
        for r in refs:
            print("    %-15s %-16s rel %.3f" % (r[0], r[1], rel(cases[r], kind, best)[0]))
        out[kind] = {"params": list(best), "refs": [list(r) for r in refs]}
    for kind, other in (("plain", "smile"), ("smile", "plain")):
        for r in out[other]["refs"]:
            r = tuple(r)
            print("control: %s rule on %s ref %s %s rel %.3f (own rule %.3f)" % (
                kind, other, r[0], r[1], rel(cases[r], kind, out[kind]["params"])[0], rel(cases[r], other, out[other]["params"])[0]))
    json.dump(out, open("close_params3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
