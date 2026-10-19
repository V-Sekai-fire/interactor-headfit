"""Desk check: the share of an eye's eyeball/iris vertices hidden from the front (a ray toward the
viewer, -Y in the fit frame, hits a skin or lid triangle in front of it), and by which role."""
import numpy as np


def hidden(c, roles, D, side=1.0):
    P = c["P"] + D
    T = c["tris"]
    eb = np.where((roles == "iris") & (np.sign(c["P"][:, 0]) == side))[0]
    cover = T[~np.isin(roles[T], ["iris"]).any(1)]
    A, B, C = P[cover[:, 0]], P[cover[:, 1]], P[cover[:, 2]]
    # 2D point-in-triangle in x,z, and the hit's y in front (more negative) of the point
    hit = np.zeros(len(eb), bool)
    by = np.full(len(eb), "", dtype=object)
    for n, v in enumerate(eb):
        p = P[v]
        x, z = p[0], p[2]
        d = (B[:, 2] - C[:, 2]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 2] - C[:, 2])
        ok = np.abs(d) > 1e-12
        l1 = np.where(ok, ((B[:, 2] - C[:, 2]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (z - C[:, 2])) / np.where(ok, d, 1), -1)
        l2 = np.where(ok, ((C[:, 2] - A[:, 2]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (z - C[:, 2])) / np.where(ok, d, 1), -1)
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        y = l1 * A[:, 1] + l2 * B[:, 1] + l3 * C[:, 1]
        front = inside & (y < p[1] - 1e-4)
        if front.any():
            hit[n] = True
            by[n] = roles[cover[np.argmin(np.where(front, y, np.inf)), 0]]
    return hit.mean(), {k: int((by == k).sum()) for k in set(by[hit])}
