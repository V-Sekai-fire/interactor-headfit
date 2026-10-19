import pandas as pd
S = pd.read_parquet("out/eye_scores.parquet")
s = S[S.status == "scored"]
print(s[["uuid", "name", "side", "kind", "rel", "rel_other_kind", "rel_wrong_eye", "role"]].assign(uuid=s.uuid.str[:8], name=s.name.str[-28:]).to_string())
one = s[s.rel_wrong_eye.notna()]
print("one-eye refs", len(one), "beat wrong eye", (one.rel < one.rel_wrong_eye).mean().round(2))
e = S[S.status.str.startswith("rule error") | S.status.str.startswith("eyes not")]
print(e.merge(pd.read_parquet("out/splits.parquet")[["uuid", "role"]], on="uuid", suffixes=("", "_s"))[["uuid", "name", "status", "role"]].assign(name=e.name.str[-28:].values).drop_duplicates("uuid").to_string())
