"""Desk probe (not committed): a procedural anime eye close on Mire, scored against the artist wink.

squash: every vertex near the left eye surface moves vertically toward a closing line, fully
inside the eye's outline and smoothly less outside it; scored by rms(procedural - artist) over
the vertices either moves, relative to the artist's rms.
"""
import re, sys, collections
import numpy as np

D = sys.argv[1]
u = open(D + "/avatar_body.usda").read()
def arr(name):
    return re.search(name + r"\s*=\s*\[(.*?)\]", u, re.S).group(1)
idx = np.array([int(x) for x in arr(r"int\[\] faceVertexIndices").split(",")]).reshape(-1, 3)
pts = np.array([float(x) for x in re.findall(r"-?[\d.]+(?:e-?\d+)?", arr(r"point3f\[\] points"))]).reshape(-1, 3)
sub = np.array([int(x) for x in arr(r"int\[\] primvars:submesh").split(",")])
shapes = {}; cur = None
for line in open(D + "/avatar_shapes.txt"):
    if line.startswith("shape "):
        cur = line.split()[1]; shapes[cur] = {}
    elif cur:
        f = line.split(); shapes[cur][int(f[0])] = np.array([float(v) for v in f[1:4]])
def dense(n):
    a = np.zeros_like(pts)
    for i, v in shapes[n].items():
        a[i] = v
    return a

def score(pred, ref):
    m = (np.linalg.norm(pred, axis=1) > 1e-6) | (np.linalg.norm(ref, axis=1) > 1e-6)
    e = np.sqrt((np.linalg.norm(pred[m] - ref[m], axis=1) ** 2).mean())
    r = np.sqrt((np.linalg.norm(ref[m], axis=1) ** 2).mean())
    return e * 1e3, r * 1e3, e / r

def eye_close(side, eye_sub, lo_frac, fall, lower_share):
    ev = np.unique(idx[sub == eye_sub].ravel())
    ev = ev[np.sign(pts[ev, 0]) == side]
    E = pts[ev]
    x0, x1 = E[:, 0].min(), E[:, 0].max()
    # the outline's lower and upper edge as functions of x (8 bins)
    bins = np.linspace(x0, x1, 9); xs, zl, zu = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = E[(E[:, 0] >= a) & (E[:, 0] <= b)]
        if len(s):
            xs.append(s[:, 0].mean()); zl.append(s[:, 2].min()); zu.append(s[:, 2].max())
    xs, zl, zu = map(np.array, (xs, zl, zu))
    H = (zu - zl).mean()
    out = np.zeros_like(pts)
    cx = 0.5 * (x0 + x1); hw = 0.5 * (x1 - x0)
    near = np.where((np.sign(pts[:, 0]) == side) & (np.abs(pts[:, 0] - cx) < hw + fall * H) &
                    (pts[:, 2] > zl.min() - fall * H) & (pts[:, 2] < zu.max() + fall * H) &
                    (pts[:, 1] < E[:, 1].max() + 0.01))[0]
    for v in near:
        x, z = pts[v, 0], pts[v, 2]
        l = np.interp(x, xs, zl); h = np.interp(x, xs, zu)
        c = l + lo_frac * (h - l)
        # distance outside the outline, in units of H
        dx = max(0.0, abs(x - cx) - hw) / H
        dz = max(0.0, l - z, z - h) / H
        d = np.hypot(dx, dz) / fall
        w = 1.0 if d <= 0 else max(0.0, 1.0 - d) ** 2 * (1 + 2 * d) if d < 1 else 0.0
        if w <= 0:
            continue
        if z >= c:
            out[v, 2] = -(z - c) * w
        else:
            out[v, 2] = (c - z) * w * lower_share
    return out

ref = dense("face_wink_L")
print("artist face_wink_L rms over its support %.2f mm" % score(ref, ref)[1])
zero = np.zeros_like(pts)
print("control zero shape: err %.2f mm rel %.3f" % (score(zero, ref)[0], score(zero, ref)[2]))
best = None
for lo in (0.2, 0.3, 0.4, 0.5):
    for fall in (0.3, 0.6, 1.0):
        for ls in (0.3, 0.6, 1.0):
            e, r, rel = score(eye_close(+1, 2, lo, fall, ls), ref)
            if best is None or rel < best[0]:
                best = (rel, e, lo, fall, ls)
print("best squash: rel %.3f err %.2f mm (closing line %.1f of the height, falloff %.1f H, lower lid share %.1f)" % best)
# the wrong side as a planted control
e, r, rel = score(eye_close(-1, 2, best[2], best[3], best[4]), ref)
print("control right eye against the left wink: rel %.3f" % rel)
np.save(D + "/../../../../anime_close_L.npy", eye_close(+1, 2, best[2], best[3], best[4]))
