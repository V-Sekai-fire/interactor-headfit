"""End to end without names: eye_finder roles -> the close rule -> scored against each avatar's own
close (answer key only). Same rule and params as headfit's named path, minus the naming check.
  python eval_e2e.py [role]  -> out/eye_e2e_<role>.parquet"""
import pathlib
import sys

import numpy as np
import pandas as pd

AXL = pathlib.Path(__file__).parent
UE = pathlib.Path.home() / "Desktop/stk-work/ue"
sys.path.insert(0, str(AXL)); sys.path.insert(0, str(UE))
role = sys.argv[1] if len(sys.argv) > 1 else "tune"
sys.argv = sys.argv[:1]
import eye_finder  # noqa: E402
import score  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_close3h as p3  # noqa: E402
import probe_lid  # noqa: E402

p3.HULL = "lo"
PARAMS = {"plain": (0.4, 1.0, 0.0, 0.5), "smile": (0.5, 1.0, 0.2, 0.5)}


def close(c, kind, side_sign):
    cc = dict(c, left_sign=side_sign, roles2=c["roles"])
    p3.CAP = 0.0 if kind == "plain" else None
    return p3.close(cc, kind, *PARAMS[kind])


A = pd.read_parquet(AXL / "out/assets.parquet").set_index("uuid")
K = pd.read_parquet(AXL / ("out/eye_key_%s.parquet" % role))
R = pd.read_parquet(AXL / "out/eye_refs.parquet")
R = R[R.side == "L"].drop_duplicates("uuid").set_index("uuid")
M = pd.read_parquet(AXL / "out/materials.parquet")
rows = []
for k in K[K.status == "ok"].itertuples():
    u = k.uuid
    a = A.loc[u]
    path = str(AXL / ("raw/files/%s.glb" % u if a.glb_status == "original" else "glb/%s.glb" % u))
    mi, ti = int(R.mesh_ordinal[u]), int(R.target_ordinal[u])
    named = bool(M[(M.uuid == u) & (M.mesh_ordinal == mi)].role.isin(["lash", "iris"]).any())
    row = dict(uuid=u, named_materials=named)
    try:
        roles, why = eye_finder.roles(path, mi, ti, AXL / "rf" / (u + ".png"))
        if roles is None:
            rows.append(dict(row, status="not found: " + why)); continue
        c, why = score.load_case(u, path, mi, ti, roles=roles)
        if c is None:
            rows.append(dict(row, status="case: " + why)); continue
        c["sel"] = np.sign(c["P"][:, c["hz"]]) == c["left_sign"]
        kind = pc.kind_of(c)[0]
        other = "smile" if kind == "plain" else "plain"
        rel = probe_lid.score(close(c, kind, c["left_sign"]), c["ref"], c["sel"])[0]
        rows.append(dict(row, status="scored", kind=kind, rel=rel,
                         rel_other_kind=probe_lid.score(close(c, other, c["left_sign"]), c["ref"], c["sel"])[0],
                         rel_wrong_eye=probe_lid.score(close(c, kind, -c["left_sign"]), c["ref"], c["sel"])[0],
                         n_eye=int((roles == "eye").sum()), n_iris=int((roles == "iris").sum())))
    except Exception as e:
        rows.append(dict(row, status="error %s: %s" % (type(e).__name__, str(e)[:70])))
S = pd.DataFrame(rows)
S.to_parquet(AXL / ("out/eye_e2e_%s.parquet" % role), compression="zstd", index=False)
print(S.status.str.slice(0, 60).value_counts().to_string())
s = S[S.status == "scored"]
print("scored %d of %d" % (len(s), len(S)))
for (k, nm), g in s.groupby(["kind", "named_materials"]):
    print("%-5s named=%-5s n %3d  rel median %.3f  < 0.8: %3d  < 1.0: %3d  beats other kind %.2f  beats wrong eye %.2f" % (
        k, nm, len(g), g.rel.median(), (g.rel < 0.8).sum(), (g.rel < 1).sum(), (g.rel < g.rel_other_kind).mean(), (g.rel < g.rel_wrong_eye).mean()))
