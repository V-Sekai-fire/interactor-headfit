import pandas as pd
S = pd.read_parquet("out/eye_scores.parquet")
A = pd.read_parquet("out/assets.parquet").set_index("uuid")
M = pd.read_parquet("out/materials.parquet")
f = S[S.role == "frozen"].drop_duplicates("uuid")
print(f.status.value_counts().to_string())
for r in f.itertuples():
    mats = M[(M.uuid == r.uuid) & (M.mesh_ordinal == r.mesh_ordinal)]
    print(r.uuid[:8], A.loc[r.uuid].glb_status, "%-26s" % r.name[-26:], "| mats on mesh:", list(zip(mats.material.str[:22], mats.role))[:5],
          "| other meshes w/ eye mats:", int(((M.uuid == r.uuid) & (M.mesh_ordinal != r.mesh_ordinal) & M.role.isin(["eye", "iris", "lash"])).sum()))
