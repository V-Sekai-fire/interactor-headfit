"""Desk probe (not committed): Mire's eye topology and the artist wink, to design an anime blink."""
import re, sys, collections, math
import numpy as np

D = sys.argv[1]
u = open(D + "/avatar_body.usda").read()
def arr(name, conv):
    m = re.search(name + r"\s*=\s*\[(.*?)\]", u, re.S)
    return m.group(1)
idx = np.array([int(x) for x in arr(r"int\[\] faceVertexIndices", int).split(",")]).reshape(-1, 3)
pts = np.array([float(x) for x in re.findall(r"-?[\d.]+(?:e-?\d+)?", arr(r"point3f\[\] points", float))]).reshape(-1, 3)
sub = np.array([int(x) for x in arr(r"int\[\] primvars:submesh", int).split(",")])
print("V", len(pts), "T", len(idx), "submesh tris", collections.Counter(sub.tolist()))
shapes = {}
cur = None
for line in open(D + "/avatar_shapes.txt"):
    if line.startswith("shape "):
        _, n, c = line.split()
        cur = n; shapes[n] = {}
    elif cur:
        f = line.split()
        shapes[cur][int(f[0])] = np.array([float(v) for v in f[1:4]])
w = shapes["face_wink_L"]
ids = np.array(sorted(w)); d = np.array([w[i] for i in ids]); m = np.linalg.norm(d, axis=1)
big = ids[m > 0.2 * m.max()]
print("wink_L moved", len(ids), "max mm", m.max() * 1e3, ">20%", len(big))
p = pts[big]; print("wink_L big region x", p[:, 0].min(), p[:, 0].max(), "z", p[:, 2].min(), p[:, 2].max(), "y", p[:, 1].min(), p[:, 1].max())
print("mean delta of big (mm)", d[m > 0.2 * m.max()].mean(0) * 1e3)
# boundary edges per submesh around the eye
ec = collections.Counter()
for t in idx:
    for k in range(3):
        a, b = sorted((t[k], t[(k + 1) % 3]))
        ec[(a, b)] += 1
bv = set()
for (a, b), c in ec.items():
    if c == 1:
        bv.add(a); bv.add(b)
box = (pts[:, 0] > p[:, 0].min() - 0.01) & (pts[:, 0] < p[:, 0].max() + 0.01) & (pts[:, 2] > p[:, 2].min() - 0.01) & (pts[:, 2] < p[:, 2].max() + 0.01)
print("boundary verts in eye box", sum(1 for v in bv if box[v]))
# which submesh do the moved vertices touch
vs = collections.Counter()
for ti, t in enumerate(idx):
    if any(v in w for v in t):
        vs[int(sub[ti])] += 1
print("tris touching wink_L by submesh", vs)
# delta direction: vertical share
print("wink_L delta z share", np.abs(d[:, 2]).sum() / np.abs(d).sum())
for n in ["face_blink", "eye_squeeze", "eye_flatten", "eye_grow", "eye_shrink"]:
    s = shapes[n]; mm = np.array([np.linalg.norm(v) for v in s.values()])
    print(n, len(s), "max mm %.2f" % (mm.max() * 1e3))
for s in (0, 1, 2):
    vv = np.unique(idx[sub == s].ravel()); q = pts[vv]
    print("submesh", s, "verts", len(vv), "x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" % (q[:,0].min(), q[:,0].max(), q[:,1].min(), q[:,1].max(), q[:,2].min(), q[:,2].max()))
# boundary loops in the eye box by submesh
for s in (0, 1, 2):
    e2 = collections.Counter()
    for t in idx[sub == s]:
        for k in range(3):
            a, b = sorted((t[k], t[(k + 1) % 3])); e2[(a, b)] += 1
    bvs = {v for (a, b), c in e2.items() if c == 1 for v in (a, b) if box[v]}
    print("submesh", s, "own-boundary verts in eye box", len(bvs))
