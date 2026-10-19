"""Desk probe (not committed): plain and smile anime eye closes, v2, scored leave-one-avatar-out.

Kinds (references/anime-face-shapes.md): plain = only the upper lid comes down; smile = both lids
meet near the middle. Each reference shape is classified by its own lower-lid travel, never by name.

v2 rule, per x across the eye outline (lo(x), hi(x) its lower and upper edge, H its height):
  closing line c(x) = lo + k (hi - lo); plain sets the lower share to 0, smile lets it rise.
  - vertices between c and hi (the eye white, the iris's upper half, lid skin): onto c;
  - vertices above hi (lash, lid skin): moved RIGIDLY by the whole opening at that x, fading over
    `above` H, so the lash keeps its thickness and bends into the line's curve;
  - smile: vertices between lo and c rise onto c; those below lo rise rigidly by (c - lo), fading
    over `below` H;
  - iris/highlight materials never move (covered, not squashed); brows never move.
"""
import itertools, json, re, struct, sys
import numpy as np

exec(open("probe_vrm_blink.py").read().split("def study")[0])
import probe_lid  # noqa: E402  role_of, vrm_case, mire_case, score

S = "/Users/ernest.lee/Desktop/smoke-water"
GIRL = S + "/3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm"
ALICIA = S + "/4-entities/godot-vrm/vrm_samples/AliciaSolid_vrm-0.51.vrm"
CANDIDATES = [("three-vrm-girl", GIRL, n) for n in ("Fcl_EYE_Close_L", "Fcl_EYE_Joy_L")] + \
             [("AliciaSolid", ALICIA, n) for n in ("eye_winkL2", "eye_winkL", "eye_smile", "eye_blink")] + \
             [("Mire", None, n) for n in ("face_wink_L", "face_smile", "face_blink")]


def vrm_target(path, suffix):
    J, acc = load(path)
    for mi_, m in enumerate(J["meshes"]):
        names = m.get("extras", {}).get("targetNames") or (m["primitives"][0].get("extras", {}) or {}).get("targetNames") or []
        for k, n in enumerate(names):
            if n.endswith(suffix):
                return mi_, k, acc(m["primitives"][0]["targets"][k]["POSITION"])
    raise KeyError(suffix)


_base = {}


def case(avatar, path, target):
    if avatar not in _base:
        _base[avatar] = probe_lid.mire_case("dress-on/project/fixtures/headfit") if path is None else probe_lid.vrm_case(path)
        _base[avatar]["tris"] = tris_of(avatar, path)
    c = dict(_base[avatar])
    if path is None:
        ref = np.zeros_like(c["P"]); cur = None
        for line in open("dress-on/project/fixtures/headfit/avatar_shapes.txt"):
            if line.startswith("shape "):
                cur = line.split()[1]
            elif cur == target:
                f = line.split(); ref[int(f[0])] = [float(v) for v in f[1:4]]
        c["mesh"] = "Body"
    else:
        mi_, k, ref = vrm_target(path, target)
        c["mesh_index"], c["target_index"] = mi_, k
    c["ref"] = ref
    c["target"] = target
    hz = [a for a in (0, 1, 2) if a != c["up"] and abs(c["fwd"][a]) < 0.5][0]
    c["hz"] = hz
    c["sel"] = np.sign(c["P"][:, hz]) == c["left_sign"]
    return c


def tris_of(avatar, path):
    if path is None:
        u = open("dress-on/project/fixtures/headfit/avatar_body.usda").read()
        return np.array([int(x) for x in re.search(r"int\[\] faceVertexIndices\s*=\s*\[(.*?)\]", u, re.S).group(1).split(",")]).reshape(-1, 3)
    J, acc = load(path)
    mi_, k, _ = vrm_target(path, {"three-vrm-girl": "Fcl_EYE_Close_L", "AliciaSolid": "eye_winkL2"}[avatar])
    return np.concatenate([acc(pr["indices"]).reshape(-1, 3).astype(np.int64) for pr in J["meshes"][mi_]["primitives"]])


def kind_of(c):
    """smile when the lower lid rises at least a third as far as the upper lid falls."""
    d = c["ref"][c["sel"]]; up = c["up"]
    m = np.linalg.norm(d, axis=1); big = m > 0.3 * m.max()
    down = -d[big & (d[:, up] < 0), up].mean() if (big & (d[:, up] < 0)).any() else 0.0
    rise = d[big & (d[:, up] > 0), up].mean() if (big & (d[:, up] > 0)).any() else 0.0
    nr = (big & (d[:, up] > 0)).sum() / max(1, big.sum())
    return ("smile" if nr >= 0.1 and rise >= 0.5 * down else "plain"), down * 1e3, rise * 1e3, nr


def eye_roles(c):
    """Split the eye-role vertices into connected parts: a part wholly above the largest one is a
    brow, a small part (under 15% of the largest) is a highlight or an iris dot; both stay still."""
    if "roles2" in c:
        return c["roles2"]
    roles = c["roles"].copy()
    P, up = c["P"], c["up"]
    tris = c.get("tris")
    if tris is None or (roles == "lash").any() or (roles == "iris").any():
        # material names already split the eye into parts; geometry is the fallback only
        c["roles2"] = roles
        return roles
    eye = roles == "eye"
    T = tris[eye[tris].all(1)]
    par = np.arange(len(P))
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for t in T:
        for k in (1, 2):
            ra, rb = f(t[0]), f(t[k])
            par[ra] = rb
    for side in (-1.0, 1.0):
        vs = np.where(eye & (np.sign(P[:, c["hz"]]) == side))[0]
        groups = {}
        for v in vs:
            groups.setdefault(f(v), []).append(v)
        if not groups:
            continue
        big = max(groups.values(), key=len)
        btop = P[big, up].max()
        for g in groups.values():
            if g is big:
                continue
            if P[g, up].min() > btop:
                roles[g] = "brow"
            elif len(g) < 0.15 * len(big):
                roles[g] = "iris"
    c["roles2"] = roles
    return roles


def close(c, kind, k, above, below, margin):
    P, up, hz = c["P"], c["up"], c["hz"]
    roles = eye_roles(c)
    side = c["left_sign"]
    da = int(np.argmax(np.abs(c["fwd"])))
    eye = np.where((roles == "eye") & (np.sign(P[:, hz]) == side))[0]
    E = P[eye]
    x0, x1 = E[:, hz].min(), E[:, hz].max(); W = x1 - x0
    bins = np.linspace(x0, x1, 11); xs, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean()); lo.append(s[:, up].min()); hi.append(s[:, up].max())
    xs, lo, hi = map(np.array, (xs, lo, hi))
    H = float((hi - lo).max())
    front = (E[:, da] * c["fwd"][da]).min() - 0.25 * H
    out = np.zeros_like(P)
    for v in np.where(np.sign(P[:, hz]) == side)[0]:
        if roles[v] in ("iris", "brow") or (P[v, da] * c["fwd"][da]) < front:
            continue
        x, h = P[v, hz], P[v, up]
        ox = max(0.0, x0 - x, x - x1) / max(W, 1e-9)
        if ox > margin:
            continue
        wx = 1.0 if ox <= 0 else (1.0 - ox / margin) ** 2
        l = np.interp(x, xs, lo); t = np.interp(x, xs, hi)
        cl = l + k * (t - l)
        dh = 0.0
        if cl <= h <= t:
            dh = -(h - cl)
        elif t < h <= t + above * H:
            dh = -(t - cl) * (1.0 - (h - t) / (above * H)) ** 2
        elif kind == "smile" and l <= h < cl:
            dh = cl - h
        elif kind == "smile" and l - below * H <= h < l:
            dh = (cl - l) * (1.0 - (l - h) / (below * H)) ** 2
        out[v, up] = dh * wx
    return out


GRID = {"plain": list(itertools.product((0.0, 0.1, 0.2, 0.3), (0.3, 0.6, 1.0), (0.0,), (0.1, 0.25, 0.5))),
        "smile": list(itertools.product((0.3, 0.4, 0.5, 0.6), (0.3, 0.6, 1.0), (0.2, 0.4, 0.8), (0.1, 0.25, 0.5)))}


def main():
    cases = {}
    print("reference shapes and their kind (left eye; upper lid down, lower lid up, share of moving vertices going up):")
    for av, path, t in CANDIDATES:
        c = case(av, path, t)
        kd, dn, ri, nr = kind_of(c)
        c["kind"] = kd
        dup = [key for key, o in cases.items() if key[0] == av and o["ref"].shape == c["ref"].shape and np.array_equal(o["ref"], c["ref"])]
        if dup:
            print("  %-15s %-16s same vertices as %s; left out" % (av, t, dup[0][1]))
            continue
        cases[(av, t)] = c
        print("  %-15s %-16s %-6s down %5.1f mm  up %5.1f mm  up-share %.2f" % (av, t, kd, dn, ri, nr))
    result = {}
    for kind in ("plain", "smile"):
        refs = [key for key, c in cases.items() if c["kind"] == kind]
        avatars = sorted({a for a, _ in refs})
        print(f"\n== {kind}: references {refs}")
        for hold in (avatars if len(avatars) > 1 else []):
            tune = [r for r in refs if r[0] != hold]
            test = [r for r in refs if r[0] == hold]
            best = None
            for g in GRID[kind]:
                rel = max(probe_lid.score(close(cases[r], kind, *g), cases[r]["ref"], cases[r]["sel"])[0] for r in tune)
                if best is None or rel < best[0]:
                    best = (rel, g)
            for r in test:
                rel, e, mag = probe_lid.score(close(cases[r], kind, *best[1]), cases[r]["ref"], cases[r]["sel"])
                print("  hold out %-15s tuned on %s -> params %s, tuned worst %.3f; HELD-OUT %s rel %.3f (%.2f of %.2f mm)"
                      % (hold, [t for t in tune], best[1], best[0], r[1], rel, e, mag))
        best = None
        for g in GRID[kind]:
            rel = max(probe_lid.score(close(cases[r], kind, *g), cases[r]["ref"], cases[r]["sel"])[0] for r in refs)
            if best is None or rel < best[0]:
                best = (rel, g)
        print("  all %s: params %s worst rel %.3f" % (kind, best[1], best[0]))
        for r in refs:
            print("    %s %s rel %.3f" % (r[0], r[1], probe_lid.score(close(cases[r], kind, *best[1]), cases[r]["ref"], cases[r]["sel"])[0]))
        result[kind] = {"params": best[1], "refs": refs}
    # control: each kind's rule against the other kind's references must score worse than its own
    for kind, other in (("plain", "smile"), ("smile", "plain")):
        for r in result[other]["refs"]:
            rel = probe_lid.score(close(cases[r], kind, *result[kind]["params"]), cases[r]["ref"], cases[r]["sel"])[0]
            print("control: %s rule on %s reference %s %s rel %.3f" % (kind, other, r[0], r[1], rel))
    json.dump({k: {"params": list(v["params"]), "refs": [list(x) for x in v["refs"]]} for k, v in result.items()},
              open("close_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()
