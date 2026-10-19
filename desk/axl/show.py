import pandas as pd
S = pd.read_parquet("out/eye_scores.parquet")
A = pd.read_parquet("out/assets.parquet").set_index("uuid")
s = S[S.status == "scored"].copy()
s["url"] = s.uuid.map(A.url).str.replace("https://raw.githubusercontent.com/", "", regex=False).str.slice(0, 90)
for u, g in s.groupby("uuid"):
    print(u[:8], g.split.iloc[0], g.roles.iloc[0], "\n   ", g.url.iloc[0])
    for r in g.itertuples():
        print("    %-40s %-10s %-5s rel %6.3f other %6.3f wrong %6.3f" % (r.name[-40:], r.side, r.kind, r.rel, r.rel_other_kind, r.rel_wrong_eye))
print(S[S.status.str.startswith("rule error")][["uuid", "name"]].to_string())
