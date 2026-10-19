"""Re-split the Articulation-XL eye-close assets by geometry (out/splits.parquet), so a model copied
into both of the mirror's splits stays out of the frozen set.

Key per asset: the face mesh (the one holding its first eye-close target), world positions,
centred on the bounding box, scaled to unit diagonal, rounded to 1e-3, sorted, sha256. A group of
assets with one key is frozen only when every member is in the mirror's test split; a group that
spans train and test goes to tune (reason "leak: same geometry in train")."""
import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.argv = sys.argv[:1]
import score  # noqa: E402
import case2  # noqa: E402


def geometry_key(path, mi):
    J, acc = case2._load(path)
    m = J["meshes"][mi]
    seen, P = set(), []
    for pr in m["primitives"]:
        a = int(pr["attributes"]["POSITION"])
        if a not in seen:
            seen.add(a)
            P.append(acc(a))
    P = np.concatenate(P)
    nodes = [i for i, n in enumerate(J.get("nodes", [])) if n.get("mesh") == mi]
    if nodes:
        M = score.node_world(J)[nodes[0]]
        P = P @ M[:3, :3].T + M[:3, 3]
    lo, hi = P.min(0), P.max(0)
    Q = np.round((P - 0.5 * (lo + hi)) / max(np.linalg.norm(hi - lo), 1e-12), 3) + 0.0
    Q = Q[np.lexsort(Q.T[::-1])]
    return hashlib.sha256(Q.astype("<f8").tobytes()).hexdigest()[:16]


def main(plant=False):
    A = pd.read_parquet(ROOT / "out/assets.parquet").set_index("uuid")
    R = pd.read_parquet(ROOT / "out/eye_refs.parquet")
    first = R.sort_values(["uuid", "mesh_ordinal", "target_ordinal"]).drop_duplicates("uuid")
    rows = []
    for r in first.itertuples():
        a = A.loc[r.uuid]
        path = ROOT / ("raw/files/%s.glb" % r.uuid if a.glb_status == "original" else "glb/%s.glb" % r.uuid)
        try:
            k = geometry_key(str(path), r.mesh_ordinal)
        except Exception as e:
            k = "unreadable:" + r.uuid
        rows.append(dict(uuid=r.uuid, mirror_split=a.split, geometry_key=k))
    S = pd.DataFrame(rows)
    if plant:  # control: give one test asset a train asset's key; the leak count must rise by one
        t = S.index[S.mirror_split == "test"][0]
        S.loc[t, "geometry_key"] = S[S.mirror_split == "train"].geometry_key.iloc[0]
    g = S.groupby("geometry_key").mirror_split.agg(lambda s: set(s))
    S["role"] = S.geometry_key.map(lambda k: "frozen" if g[k] == {"test"} else "tune")
    S["reason"] = S.apply(lambda r: "leak: same geometry in train" if r.mirror_split == "test" and r.role == "tune" else "mirror split " + r.mirror_split, axis=1)
    S["rule"] = "geometry key v1 (bbox-centred, unit diagonal, 1e-3, sorted, sha256[:16])"
    leaks = int((S.reason.str.startswith("leak")).sum())
    if not plant:
        assert not S.duplicated("uuid").any()
        S.to_parquet(ROOT / "out/splits.parquet", compression="zstd", index=False)
    print("%sassets %d, geometry groups %d, test -> frozen %d, test leaked into tune %d" % (
        "PLANTED " if plant else "", len(S), S.geometry_key.nunique(), int((S.role == "frozen").sum()), leaks))
    return leaks


if __name__ == "__main__":
    base = main()
    assert main(plant=True) == base + 1, "planted leak not caught"
    print("planted leak caught")
