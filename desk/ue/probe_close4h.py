"""Desk probe (not committed): close v4 = v3, but when the eye parts came from geometry (no lash or
iris named), the lid part (the largest eye part: Mire's lash line) slides down in one piece instead
of being squashed onto the closing line: dz = f * median over the part's columns of (cl(x) - its
lowest z there), so the lid keeps its arc (Mire's own closes move it -12..-16 mm, near uniform)."""
import sys
import numpy as np
sys.argv = sys.argv[:1]
import probe_close3h as p3  # noqa: E402


CAPS = {"plain": True, "smile": False}


def close(c, kind, k, above, below, margin, arch=0.0, f=1.0, rigid=True):
    p3.CAP = 0.0 if CAPS.get(kind) else None
    out = p3.close(c, kind, k, above, below, margin, arch)
    roles = p3.eye_roles(c)
    named = (c["roles"] == "lash").any() or (c["roles"] == "iris").any()
    if named or not rigid:
        return out
    P, up, hz = c["P"], c["up"], c["hz"]
    side = np.sign(P[:, hz]) == c["left_sign"]
    lid = np.where(side & (roles == "eye"))[0]
    E = P[side & np.isin(roles, ["eye", "iris"])]
    x0, x1 = E[:, hz].min(), E[:, hz].max()
    bins = np.linspace(x0, x1, 11)
    IX, IL = [], []
    I = P[side & (roles == "iris")]
    for a, b in zip(bins[:-1], bins[1:]):
        Ib = I[(I[:, hz] >= a) & (I[:, hz] <= b)]
        if len(Ib):
            IX.append(Ib[:, hz].mean()); IL.append(Ib[:, up].min())
    gaps, lx, ll = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        L = P[lid][(P[lid, hz] >= a) & (P[lid, hz] <= b)]
        if len(s) and len(L):
            lo, hi = s[:, up].min(), s[:, up].max()
            if p3.HULL:
                hl, hh = (float(e[0]) for e in p3.hull_edges(E[:, hz], E[:, up], np.array([s[:, hz].mean()])))
                lo = hl
                hi = hh if p3.HULL is True else hi
            clx = lo + k * (hi - lo)
            clx = min(clx, hi)
            if CAPS.get(kind) and IX:
                clx = max(min(clx, float(np.interp(s[:, hz].mean(), IX, IL))), lo)
            gaps.append(clx - L[:, up].min())
            lx.append(L[:, hz].mean()); ll.append(L[:, up].min())
    shift = f * float(np.quantile(gaps, Q))
    if shift >= 0:
        # The largest part reaches the eye's lower edge: it is the eye white (Kimono), not a lash
        # line above the eyeball (Mire). The squashing rule stands.
        return out
    if FOLLOW and shift < 0:
        # The lid skin at and above the lash line moves with it, not past it (it would roll over
        # the lash); the skin below the lash still closes down to the closing line.
        o = np.argsort(lx)
        at = np.interp(P[:, hz], np.array(lx)[o], np.array(ll)[o])
        dn = (out[:, up] < 0) & (P[:, up] >= at)
        out[dn, up] = np.maximum(out[dn, up], shift)
    out[lid, up] = shift
    return out


FOLLOW = True
Q = 0.3
