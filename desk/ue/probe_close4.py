"""Desk probe (not committed): close v4 = v3, but when the eye parts came from geometry (no lash or
iris named), the lid part (the largest eye part: Mire's lash line) slides down in one piece instead
of being squashed onto the closing line: dz = f * median over the part's columns of (cl(x) - its
lowest z there), so the lid keeps its arc (Mire's own closes move it -12..-16 mm, near uniform)."""
import sys
import numpy as np
sys.argv = sys.argv[:1]
import probe_close3 as p3  # noqa: E402


def close(c, kind, k, above, below, margin, arch=0.0, f=1.0, rigid=True):
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
    gaps = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, hz] >= a) & (E[:, hz] <= b)]
        L = P[lid][(P[lid, hz] >= a) & (P[lid, hz] <= b)]
        if len(s) and len(L):
            lo, hi = s[:, up].min(), s[:, up].max()
            gaps.append(lo + k * (hi - lo) - L[:, up].min())
    out[lid, up] = f * float(np.median(gaps))
    return out
