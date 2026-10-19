"""Desk probe (not committed): how each connected eye part moves in Mire's own smile closes, so the
smile rule knows which parts slide, which stay and which hide."""
import numpy as np
import probe_close2 as pc

for t in ("face_smile", "face_wink_L", "face_blink"):
    c = pc.case("Mire", None, t)
    roles = pc.eye_roles(c)
    P, d, up, hz = c["P"], c["ref"], c["up"], c["hz"]
    tris = c["tris"]
    eye = np.isin(c["roles"], ["eye"])
    par = np.arange(len(P))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for tr in tris[eye[tris].all(1)]:
        par[f(tr[0])] = f(tr[1]); par[f(tr[2])] = f(tr[1])
    left = eye & (np.sign(P[:, hz]) == c["left_sign"])
    groups = {}
    for v in np.where(left)[0]:
        groups.setdefault(f(v), []).append(v)
    print("==", t)
    for g in sorted(groups.values(), key=len, reverse=True):
        g = np.array(g)
        m = np.linalg.norm(d[g], axis=1)
        mv = m > 1e-5
        print("  part %4d verts role %-5s z %.3f..%.3f  moved %4d  dz down %5.1f mm (%d) up %5.1f mm (%d)  max %.1f mm" % (
            len(g), roles[g[0]], P[g, up].min(), P[g, up].max(), mv.sum(),
            -d[g][d[g][:, up] < 0, up].mean() * 1e3 if (d[g][:, up] < 0).any() else 0, (d[g][:, up] < -1e-5).sum(),
            d[g][d[g][:, up] > 0, up].mean() * 1e3 if (d[g][:, up] > 0).any() else 0, (d[g][:, up] > 1e-5).sum(), m.max() * 1e3))
    sk = (c["roles"] == "skin") & (np.sign(P[:, hz]) == c["left_sign"]) & (np.linalg.norm(d, axis=1) > 1e-5)
    print("  skin moved %d, dz mean %+.1f mm" % (sk.sum(), d[sk, up].mean() * 1e3 if sk.any() else 0))
