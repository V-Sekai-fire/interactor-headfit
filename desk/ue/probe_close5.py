"""Desk probe (not committed): close v5 = the named-eyes close (probe_close3h, lower hull, iris cap
on plain) with the NAMED lash slid in one piece per column instead of squashed by height.

Squashing by height moves a lash card's top far more than its bottom, so the lash line curls
(textured, alpha-cutout sheets: three-vrm-girl's hook, Twist's U). Avatars' own closes move the
lash nearly uniformly (three-vrm-girl 218/221 lash vertices, mean 9.5 mm down; Twist 16.0;
AliciaSolid 17.8). Here every lash vertex moves by its column's shift: the closing line minus the
upper lash's lowest point in that column (smoothed across columns), times LASH_F; lash vertices
below the closing line (the lower lash line of a lash material that rings the eye) keep the base
rule. Skin above the lash
moves at least as far as the lash below it fades out (it must not open a gap above the lash)."""
import sys

import numpy as np

sys.argv = sys.argv[:1]
import probe_close3h as p3  # noqa: E402

p3.HULL = "lo"
LASH_F = 1.0
FOLLOW = True
PARAMS = {"plain": (0.4, 1.0, 0.0, 0.5), "smile": (0.5, 1.0, 0.2, 0.5)}


def closing_line(c, kind, k, above, below, margin):
    """The close's own per-column closing line cl(x) (same construction as p3.close)."""
    P, up, hz = c["P"], c["up"], c["hz"]
    roles = c["roles"]
    side = c["left_sign"]
    E = P[np.isin(roles, ["eye", "iris"]) & (np.sign(P[:, hz]) == side)]
    x0, x1 = E[:, hz].min(), E[:, hz].max()
    bins = np.linspace(x0, x1, 11)
    xs = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean())
    xs = np.array(xs)
    lo, hi = p3.hull_edges(E[:, hz], E[:, up], xs)
    hi2 = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            hi2.append(s[:, up].max())
    hi = np.array(hi2)
    cl = np.minimum(lo + k * (hi - lo), hi)
    if kind == "plain":
        I = P[(roles == "iris") & (np.sign(P[:, hz]) == side)]
        if len(I):
            ix, il = [], []
            for a, b in zip(bins[:-1], bins[1:]):
                s = I[(I[:, hz] >= a) & (I[:, hz] <= b)]
                if len(s):
                    ix.append(s[:, hz].mean()); il.append(s[:, up].min())
            cl = np.maximum(np.minimum(cl, np.interp(xs, ix, il)), lo)
    return xs, cl, bins


LOWER_F = 0.5
EYE_STILL = False
EYE_FOLLOW = False
SLITS = False
ANCHOR = None
UNIFORM = None
CONNECTED = False
SMOOTH = 0
UPPER = 0.6  # the upper lash: lash vertices above this fraction of the eye's height


def edges(c):
    P, up, hz = c["P"], c["up"], c["hz"]
    E = P[np.isin(c["roles"], ["eye", "iris"]) & (np.sign(P[:, hz]) == c["left_sign"])]
    bins = np.linspace(E[:, hz].min(), E[:, hz].max(), 11)
    xs, lo, hi = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        if len(s):
            xs.append(s[:, hz].mean()); lo.append(s[:, up].min()); hi.append(s[:, up].max())
    return (lambda x: np.interp(x, xs, lo)), (lambda x: np.interp(x, xs, hi))


WELD = 1e-5  # metres; vertices nearer than this move together


def weld(P, out):
    """Vertices at one position (a seam between materials, or pieces modelled a hair apart) move
    together, by the member that moves furthest: a still eye-white vertex must not hold back the
    lid skin it touches, and a crease a hair wide must not open into a hole."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(WELD, output_type="ndarray")
    par = np.arange(len(P))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for x, y in pairs:
        par[f(x)] = f(y)
    root = np.array([f(i) for i in range(len(P))]) if len(pairs) else par
    order = np.argsort(root, kind="stable")
    g = root[order]
    starts = np.r_[0, np.where(np.diff(g))[0] + 1, len(g)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a > 1:
            ids = order[a:b]
            out[ids] = out[ids[np.argmax(np.linalg.norm(out[ids], axis=1))]]
    return out


PRIORITY = ["lash", "skin", "eye", "brow", "iris"]  # the role a welded seam vertex takes


def welded(c):
    """The case with vertices at one position merged into one (the exporter splits them per
    material or UV seam; nothing animates them apart): positions, roles and triangles on the
    merged vertices, and inv mapping each original vertex to its merged one."""
    key = np.round(c["P"] / 1e-5).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    rank = np.array([PRIORITY.index(r) if r in PRIORITY else len(PRIORITY) for r in c["roles"]])
    best = np.full(len(first), 99)
    np.minimum.at(best, inv, rank)
    roles = np.array([PRIORITY[b] if b < len(PRIORITY) else "skin" for b in best], dtype=object)
    tris = inv[c["tris"]]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    w = dict(c, P=c["P"][first], roles=roles, tris=tris, ref=c["ref"][first])
    w["sel"] = np.sign(w["P"][:, w["hz"]]) == w["left_sign"]
    w.pop("roles2", None)
    return w, inv


def smooth(c, out, n):
    """n passes of neighbour averaging of the motion over the connected mesh, on skin and eye
    vertices that move or touch one that does (lash, iris and brow keep their motion): the lid
    moves as one sheet, so an edge of it cannot stretch open."""
    import scipy.sparse as sp
    t = c["tris"]
    N = len(c["P"])
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(N, N))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    free = np.isin(c["roles"], ["skin", "eye"])
    moving = np.linalg.norm(out, axis=1) > 1e-6
    free &= moving | (A @ moving.astype(float) > 0)
    for _ in range(n):
        avg = (A @ out) / np.maximum(deg, 1)[:, None]
        out = np.where(free[:, None], 0.5 * out + 0.5 * avg, out)
    return out


def close(c, kind, params=None):
    if CONNECTED:
        w, inv = welded(c)
        out = _close_on(w, kind, params)
        if SMOOTH:
            out = smooth(w, out, SMOOTH)
        return out[inv]
    return _close_on(c, kind, params)


def _close_on(c, kind, params=None):
    out = _close(c, kind, params)
    if EYE_FOLLOW:
        out = eye_follow(c, out)
    out = weld(c["P"], out)
    if ANCHOR:
        out = anchor(c, out)
    if SLITS:
        out = slits(c, out)
    return out


def anchor(c, out):
    """An open edge of the skin above the eye (a lid patch tucked under the brow, twintail) stays
    put: moving it uncovers the gap the brow hid. Skin vertices on a boundary of the (welded) skin
    surface, more than ANCHOR eye heights above the eye's top, do not move."""
    P, up, hz = c["P"], c["up"], c["hz"]
    tris = c["tris"]
    key = np.round(P / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    skin = c["roles"] == "skin"
    t = inv[tris[skin[tris].all(1)]]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = np.zeros(inv.max() + 1, bool)
    bnd[u[cnt == 1].ravel()] = True
    for sgn in (1.0, -1.0):
        cc = dict(c, left_sign=sgn)
        lo, hi = edges(cc)
        E = P[np.isin(c["roles"], ["eye", "iris"]) & (np.sign(P[:, hz]) == sgn)]
        H = float(np.ptp(E[:, up]))
        m = skin & bnd[inv] & (np.sign(P[:, hz]) == sgn) & (P[:, up] > hi(P[:, hz]) + ANCHOR * H)
        out[m] = 0.0
    return out


def slits(c, out):
    """A slit in the lid skin above the eye (twintail's crease, open in the rest pose and shut in its
    own closes) must not open wider: every vertex on the slit's rim moves as its furthest-moving
    rim vertex. A slit is a hole in the skin (a boundary loop of skin-only triangles) whose rim lies
    wholly above the eye's top edge and that the close moves."""
    P, up, hz = c["P"], c["up"], c["hz"]
    tris = c["tris"]
    skin = c["roles"] == "skin"
    t = tris[skin[tris].all(1)]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    b = u[cnt == 1]
    if not len(b):
        return out
    par = {}

    def f(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for x, y in b:
        par[f(x)] = f(y)
    loops = {}
    for v in np.unique(b):
        loops.setdefault(f(v), []).append(v)
    _, hi = edges(c)
    for sgn in (1.0, -1.0):
        cc = dict(c, left_sign=sgn)
        _, hi = edges(cc)
        for L in loops.values():
            L = np.array(L)
            if not (np.sign(P[L, hz]) == sgn).all() or not (P[L, up] > hi(P[L, hz])).all():
                continue
            m = np.linalg.norm(out[L], axis=1)
            if m.max() > 1e-6:
                out[L] = out[L[np.argmax(m)]]
    return out


def eye_follow(c, out):
    """Eye-white vertices move as the lid vertex nearest them seen from the front (skin or lash
    of the same side): the white is drawn on with the lid (twintail's crease decal, three-vrm-girl's
    white rim) and must neither stay behind as a white flake nor be squashed out under the lash."""
    from scipy.spatial import cKDTree
    P, up, hz = c["P"], c["up"], c["hz"]
    for sgn in (1.0, -1.0):
        side = np.sign(P[:, hz]) == sgn
        eye = np.where(side & (c["roles"] == "eye"))[0]
        lid = np.where(side & np.isin(c["roles"], ["skin", "lash"]))[0]
        if len(eye) and len(lid):
            _, j = cKDTree(P[lid][:, [hz, up]]).query(P[eye][:, [hz, up]])
            out[eye] = out[lid[j]]
    return out


def _close(c, kind, params=None):
    params = params or PARAMS[kind]
    c = dict(c)
    c["roles2"] = c["roles"]
    p3.CAP = 0.0 if kind == "plain" else None
    out = p3.close(c, kind, *params)
    if EYE_STILL:
        out[c["roles"] == "eye"] = 0.0
    P, up, hz = c["P"], c["up"], c["hz"]
    side = np.sign(P[:, hz]) == c["left_sign"]
    xs, cl, bins = closing_line(c, kind, *params)
    # The upper lash: lash vertices above the closing line. A lash material often rings the whole
    # eye (VRoid's FaceEyeline); its lower part is the lower lash line and stays with the lower lid.
    lash_all = side & (c["roles"] == "lash")
    lo_x, hi_x = edges(c)
    band = lo_x(P[:, hz]) + UPPER * (hi_x(P[:, hz]) - lo_x(P[:, hz]))
    lash = np.where(lash_all & (P[:, up] > band))[0]
    if not len(lash):
        return out
    lx, sh, lt = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        L = lash[(P[lash, hz] >= a) & (P[lash, hz] <= b)]
        if len(L):
            x = P[L, hz].mean()
            lx.append(x); sh.append(float(np.interp(x, xs, cl)) - P[L, up].min()); lt.append(P[L, up].max())
    lx, sh = np.array(lx), np.array(sh)
    if len(sh) >= 3:
        sh = np.convolve(np.pad(sh, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    sh = np.minimum(sh, 0.0) * LASH_F  # never up
    if UNIFORM:
        sh = np.full_like(sh, float(np.quantile(sh, UNIFORM)))  # one rigid slide for the whole lash
    s_at = np.interp(P[:, hz], lx, sh)
    out[lash, up] = s_at[lash]
    if kind == "smile":
        # The lower lash line rises in one piece too (squashed onto the closing line it collapses
        # into a blob): its top goes LOWER_F of the way up to the closing line.
        low = np.where(lash_all & (P[:, up] <= band))[0]
        bx, bs = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            L = low[(P[low, hz] >= a) & (P[low, hz] <= b)]
            if len(L):
                x = P[L, hz].mean()
                bx.append(x); bs.append(max(float(np.interp(x, xs, cl)) - P[L, up].max(), 0.0) * LOWER_F)
        if len(low) and bx:
            out[low, up] = np.interp(P[low, hz], bx, bs)
    if not FOLLOW:
        return out
    # Skin and eyeline above the lash: no less motion than the lash under it, fading over the lash's
    # own height above its top (so the lid does not open a gap above the lash).
    top = np.interp(P[:, hz], lx, np.array(lt))
    Lh = max(float(np.ptp(P[lash, up])) / 2, 1e-6)
    near = side & ~np.isin(c["roles"], ["lash", "iris", "brow"]) & (P[:, up] >= top - Lh) & (P[:, up] <= top + Lh)
    fade = np.clip(1.0 - (P[:, up] - top) / Lh, 0.0, 1.0)
    want = s_at * fade
    xin = (P[:, hz] >= lx.min() - 0.1 * np.ptp(lx)) & (P[:, hz] <= lx.max() + 0.1 * np.ptp(lx))
    m = near & xin
    out[m, up] = np.minimum(out[m, up], want[m])
    return out
