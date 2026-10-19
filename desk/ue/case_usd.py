"""Desk loader (not committed): one Mesh of a Blender-exported .usda as a probe case (Z up, -Y
forward, left +X, like Mire's fixture). Faces fan-triangulated; roles from GeomSubset material names."""
import re
import numpy as np
import probe_lid

_cache = {}


def _block(u, start):
    i = u.index("{", start); d = 0
    for j in range(i, len(u)):
        d += {"{": 1, "}": -1}.get(u[j], 0)
        if d == 0:
            return u[i:j]


def _nums(s, t=float):
    return np.array([t(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s)])


def case(name, path, mesh, target):
    key = (path, mesh)
    if key not in _cache:
        u = open(path).read()
        b = _block(u, u.index('def Mesh "%s"' % mesh))
        arr = lambda n, s=b: re.search(n + r"\s*=\s*\[(.*?)\]", s, re.S).group(1)  # noqa: E731
        P = _nums(arr(r"point3f\[\] points")).reshape(-1, 3)
        cnt = _nums(arr(r"int\[\] faceVertexCounts"), int)
        idx = _nums(arr(r"int\[\] faceVertexIndices"), int)
        off = np.concatenate([[0], np.cumsum(cnt)])
        tris, tface = [], []
        for f in range(len(cnt)):
            v = idx[off[f]:off[f + 1]]
            for k in range(1, len(v) - 1):
                tris.append((v[0], v[k], v[k + 1])); tface.append(f)
        tris, tface = np.array(tris), np.array(tface)
        roles = np.array(["skin"] * len(P), dtype=object)
        for m in re.finditer(r'def GeomSubset "([^"]+)"', b):
            sb = _block(b, m.start())
            r = probe_lid.role_of(m.group(1))
            if r != "skin":
                faces = _nums(arr(r"int\[\] indices", sb), int)
                roles[np.unique(tris[np.isin(tface, faces)].ravel())] = r
        shapes = {}
        for m in re.finditer(r'def BlendShape "([^"]+)"', b):
            sb = _block(b, m.start())
            shapes[m.group(1)] = sb
        _cache[key] = (P, tris, roles, shapes)
    P, tris, roles, shapes = _cache[key]
    sb = shapes[target]
    o = _nums(re.search(r"offsets\s*=\s*\[(.*?)\]", sb, re.S).group(1)).reshape(-1, 3)
    pi = _nums(re.search(r"pointIndices\s*=\s*\[(.*?)\]", sb, re.S).group(1), int)
    ref = np.zeros_like(P); ref[pi] = o
    s = 0.07  # the rig's Body xformOp:scale
    c = dict(name=name, P=P * s, ref=ref * s, roles=roles, up=2, fwd=np.array([0, -1.0, 0]), left_sign=1.0, hz=0,
             tris=tris, target=target)
    c["sel"] = np.sign(P[:, 0]) == 1.0
    return c
