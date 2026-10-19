"""Coverage of the eye-close references: which have named eye materials on the shape's mesh."""
import pandas as pd

A = pd.read_parquet("out/assets.parquet")
M = pd.read_parquet("out/materials.parquet")
R = pd.read_parquet("out/eye_refs.parquet")
T = pd.read_parquet("out/morph_targets.parquet")
print(A.glb_status.value_counts().to_string())
k = R[["uuid", "mesh_ordinal"]].drop_duplicates()
roles = M.groupby(["uuid", "mesh_ordinal"]).role.agg(lambda s: set(s)).reset_index()
j = k.merge(roles, on=["uuid", "mesh_ordinal"], how="left")
j["has_eye"] = j.role.apply(lambda s: isinstance(s, set) and bool({"eye", "iris", "lash"} & s))
j["has_skin"] = j.role.apply(lambda s: isinstance(s, set) and "skin" in s)
u = j.groupby("uuid").agg(eye=("has_eye", "any"), skin=("has_skin", "any")).reset_index().merge(A[["uuid", "split"]], on="uuid")
print(pd.crosstab([u.split, u.eye], u.skin))


def fam(n):
    n = n.lower()
    for f in ("fcl_eye", "ak_0", "eyeblink", "vrc", "blink", "wink", "close", "joy", "smile", "まばたき"):
        if f in n:
            return f
    return "other"


R2 = R.merge(T, on=["uuid", "mesh_ordinal", "target_ordinal"])
R2["fam"] = R2.name.map(fam)
print(R2.groupby("fam").uuid.nunique().sort_values(ascending=False).to_string())
s = T[T.uuid.isin(R.uuid)].groupby("uuid").name.agg(lambda x: "|".join(sorted(x)))
print("distinct target-name sets among eye-ref assets:", s.nunique(), "of", len(s))
print(s.value_counts().head(5).rename(lambda x: x[:80]).to_string())
