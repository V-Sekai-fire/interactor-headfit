"""Desk probe (not committed): Mire's left eye opening and how the artist wink closes it."""
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
W = dense("face_wink_L")

def loops(tris):
    ec = collections.Counter()
    for t in tris:
        for k in range(3):
            ec[tuple(sorted((t[k], t[(k + 1) % 3])))] += 1
    adj = collections.defaultdict(list)
    for (a, b), c in ec.items():
        if c == 1:
            adj[a].append(b); adj[b].append(a)
    seen = set(); out = []
    for s in adj:
        if s in seen:
            continue
        comp = []; st = [s]
        while st:
            v = st.pop()
            if v in seen:
                continue
            seen.add(v); comp.append(v); st.extend(adj[v])
        out.append(comp)
    return out

skin = idx[sub == 1]
L = loops(skin)
print("skin boundary loops", len(L), sorted(len(l) for l in L)[-8:])
for l in sorted(L, key=len)[-8:]:
    c = pts[l].mean(0)
    print("loop n %d centre x %.3f y %.3f z %.3f  z span %.1f mm  wink moves mean %.2f mm" % (len(l), *c, (pts[l][:, 2].max() - pts[l][:, 2].min()) * 1e3, np.linalg.norm(W[l], axis=1).mean() * 1e3))
eyes = idx[sub == 2]
for l in sorted(loops(eyes), key=len)[-4:]:
    c = pts[l].mean(0)
    print("eye-surface loop n %d centre x %.3f y %.3f z %.3f" % (len(l), *c))
# the left eye opening: the skin loop with centre x > 0 at eye height with the most wink motion
cand = [l for l in L if pts[l].mean(0)[0] > 0.01 and 1.55 < pts[l].mean(0)[2] < 1.63]
lo = max(cand, key=lambda l: np.linalg.norm(W[l], axis=1).mean())
P = pts[lo]; x = P[:, 0]; z = P[:, 2]
cz = z.mean()
up = [v for v in lo if pts[v, 2] > cz]; dn = [v for v in lo if pts[v, 2] <= cz]
print("left opening: %d verts, width %.1f mm, height %.1f mm" % (len(lo), (x.max() - x.min()) * 1e3, (z.max() - z.min()) * 1e3))
def zlow(xx):
    q = sorted(dn, key=lambda v: pts[v, 0])
    return np.interp(xx, pts[q, 0], pts[q, 2])
r = []
for v in up:
    h = pts[v, 2] - zlow(pts[v, 0])
    if h > 0.002:
        r.append((W[v, 2] / -h, W[v, 1] * 1e3, h * 1e3))
r = np.array(r)
print("upper lid: wink dz / opening height mean %.2f (min %.2f max %.2f), dy mean %.2f mm, opening mean %.1f mm" % (r[:, 0].mean(), r[:, 0].min(), r[:, 0].max(), r[:, 1].mean(), r[:, 2].mean()))
dnm = np.array([W[v] for v in dn]) * 1e3
print("lower lid: wink mean delta (mm) x %.2f y %.2f z %.2f" % tuple(dnm.mean(0)))
np.save(D + "/../../../../left_loop.npy", np.array(lo))
