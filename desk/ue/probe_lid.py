"""Desk probe (not committed): an anime upper-lid blink, scored against each avatar's own blink.

The rule (from the anime blink study, references/anime-face-shapes.md): the upper lid group slides down
onto a closing line just above the eye's lower edge, along the eye surface; the lower half, the iris and
the highlight stay still; skin and lashes above the eye follow with a falloff.

  python3 probe_lid.py            # tune on the tuning set, score the held-out set
"""
import json, re, struct, itertools, sys
import numpy as np

exec(open("probe_vrm_blink.py").read().split("def study")[0])  # load(), blink_binds()

ROLE = [("iris", r"iris|pupil|hitomi|highlight|hilight"), ("lash", r"lash|matsuge|mastuge|eyeline|eye_line"),
        ("brow", r"brow|mayu"), ("eye", r"eye|white")]

def role_of(name):
    for r, rx in ROLE:
        if re.search(rx, name, re.I):
            return r
    return "skin"

def vrm_case(path):
    J, acc = load(path)
    ver, side, binds = blink_binds(J)
    mesh, ti, w = binds[0]
    prims = J["meshes"][mesh]["primitives"]
    pos = acc(prims[0]["attributes"]["POSITION"])
    ref = np.zeros_like(pos)
    roles = np.array(["skin"] * len(pos), dtype=object)
    for m2, t2, w2 in binds:
        if m2 == mesh:
            ref += acc(prims[0]["targets"][t2]["POSITION"]) * w2
    for pr in prims:
        assert pr["attributes"]["POSITION"] == prims[0]["attributes"]["POSITION"], "primitives must share positions"
        used = np.unique(acc(pr["indices"]).ravel().astype(np.int64))
        roles[used] = role_of(J["materials"][pr["material"]]["name"])
    # frame: up = +Y; forward = -Z (0.x) or +Z (1.0); the avatar's left = -X (0.x) or +X (1.0)
    fwd = np.array([0, 0, -1.0]) if ver == "0.x" else np.array([0, 0, 1.0])
    left = -1.0 if ver == "0.x" else 1.0
    return dict(name=path.split("/")[-1], P=pos, ref=ref, roles=roles, up=1, fwd=fwd, left_sign=left, both=(side == "both"))

def mire_case(D):
    u = open(D + "/avatar_body.usda").read()
    arr = lambda n: re.search(n + r"\s*=\s*\[(.*?)\]", u, re.S).group(1)
    idx = np.array([int(x) for x in arr(r"int\[\] faceVertexIndices").split(",")]).reshape(-1, 3)
    P = np.array([float(x) for x in re.findall(r"-?[\d.]+(?:e-?\d+)?", arr(r"point3f\[\] points"))]).reshape(-1, 3)
    sub = np.array([int(x) for x in arr(r"int\[\] primvars:submesh").split(",")])
    roles = np.array(["skin"] * len(P), dtype=object)
    roles[np.unique(idx[sub == 2].ravel())] = "eye"
    ref = np.zeros_like(P); cur = None
    for line in open(D + "/avatar_shapes.txt"):
        if line.startswith("shape "):
            cur = line.split()[1]
        elif cur == "face_wink_L":
            f = line.split(); ref[int(f[0])] = [float(v) for v in f[1:4]]
    # Z up, -Y forward, left = +X
    return dict(name="Mire", P=P, ref=ref, roles=roles, up=2, fwd=np.array([0, -1.0, 0]), left_sign=1.0, both=False)

def score(pred, ref, sel):
    m = sel & ((np.linalg.norm(pred, axis=1) > 1e-7) | (np.linalg.norm(ref, axis=1) > 1e-7))
    e = np.sqrt((np.linalg.norm(pred[m] - ref[m], axis=1) ** 2).mean())
    r = np.sqrt((np.linalg.norm(ref[m], axis=1) ** 2).mean())
    return e / r, e * 1e3, r * 1e3

def lid_close(c, k, above, margin, follow):
    """k: closing line height as a share of the eye's height; above: how far over the eye's top
    (in eye heights) skin and lashes follow; margin: sideways reach past the eye's corners (eye widths);
    follow: 1 slides along the eye surface's depth, 0 moves straight down."""
    P, roles, up = c["P"], c["roles"], c["up"]
    side = c["left_sign"]
    hz = [a for a in (0, 1, 2) if a != up and abs(c["fwd"][a]) < 0.5][0]  # the sideways axis
    da = int(np.argmax(np.abs(c["fwd"])))  # the depth axis
    eye = np.where((roles == "eye") & (np.sign(P[:, hz]) == side))[0]
    E = P[eye]
    x0, x1 = E[:, hz].min(), E[:, hz].max(); W = x1 - x0
    nb = 10
    bins = np.linspace(x0, x1, nb + 1); xs, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean()); lo.append(s[:, up].min()); hi.append(s[:, up].max())
    xs, lo, hi = map(np.array, (xs, lo, hi))
    H = float((hi - lo).max())
    front = E[:, da] * c["fwd"][da]
    out = np.zeros_like(P)
    for v in np.where(np.sign(P[:, hz]) == side)[0]:
        if roles[v] in ("iris", "brow"):
            continue
        x, h = P[v, hz], P[v, up]
        ox = max(0.0, x0 - x, x - x1) / max(W, 1e-9)
        if ox > margin:
            continue
        l = np.interp(x, xs, lo); t = np.interp(x, xs, hi)
        cl = l + k * (t - l)
        if h <= cl or h > t + above * H:
            continue
        # depth: only the front layer of the face around the eye (skip the back of the head)
        if (P[v, da] * c["fwd"][da]) < front.min() - 0.25 * H:
            continue
        wx = 1.0 if ox <= 0 else (1.0 - ox / margin) ** 2
        if h <= t:
            dh = -(h - cl)  # inside the eye: onto the closing line
            wv = 1.0
        else:
            dh = -(t - cl)  # above the eye: the whole opening, fading with height
            s = (h - t) / max(above * H, 1e-9)
            wv = (1.0 - s) ** 2
        w = wx * wv
        out[v, up] = dh * w
        if follow > 0:
            # depth of the eye surface at the old and the new height, from the nearest eye vertices
            def depth_at(hh):
                d2 = (E[:, hz] - x) ** 2 + (E[:, up] - hh) ** 2
                return E[np.argsort(d2)[:4], da].mean()
            out[v, da] = follow * w * (depth_at(h + dh) - depth_at(h))
    return out

if __name__ == "__main__":
    S = "/Users/ernest.lee/Desktop/smoke-water"
    cases = {
        "three-vrm-girl": vrm_case(S + "/3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm"),
        "AliciaSolid": vrm_case(S + "/4-entities/godot-vrm/vrm_samples/AliciaSolid_vrm-0.51.vrm"),
        "Mire": mire_case("dress-on/project/fixtures/headfit"),
    }
    tune, hold = ["three-vrm-girl", "AliciaSolid"], ["Mire"]
    for n, c in cases.items():
        side = np.sign(c["P"][:, [a for a in (0, 1, 2) if a != c["up"] and abs(c["fwd"][a]) < 0.5][0]]) == c["left_sign"]
        c["sel"] = side
        print("%-15s eye vertices %d, ref rms %.2f mm, zero shape rel %.3f" % (n, ((c["roles"] == "eye") & side).sum(), score(np.zeros_like(c["P"]), c["ref"], side)[2], score(np.zeros_like(c["P"]), c["ref"], side)[0]))
    grid = list(itertools.product((0.0, 0.1, 0.2, 0.3), (0.3, 0.6, 1.0), (0.1, 0.25), (0.0, 1.0)))
    res = []
    for g in grid:
        rel = [score(lid_close(cases[n], *g), cases[n]["ref"], cases[n]["sel"])[0] for n in tune]
        res.append((max(rel), g, rel))
    res.sort()
    best = res[0]
    print("tuned on %s: k %.1f above %.1f margin %.2f follow %.0f -> rel %s (worst %.3f)" % (tune, *best[1], ["%.3f" % r for r in best[2]], best[0]))
    for n in tune + hold:
        rel, e, r = score(lid_close(cases[n], *best[1]), cases[n]["ref"], cases[n]["sel"])
        print("%-15s %s rel %.3f  error %.2f mm of %.2f mm" % (n, "HELD OUT" if n in hold else "tuned   ", rel, e, r))
    # control: the same rule on the wrong side must not match the left blink
    for n in tune + hold:
        c = dict(cases[n]); c["left_sign"] = -c["left_sign"]
        rel = score(lid_close(c, *best[1]), cases[n]["ref"], np.ones(len(c["P"]), bool))[0]
        print("%-15s control (right eye vs the left blink) rel %.3f" % (n, rel))
