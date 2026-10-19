"""Hand labels (by eye, from rf_sheets) and the finder's end-to-end score on the anime faces only."""
import pathlib
import pandas as pd
AXL = pathlib.Path(__file__).parent
ANIME = [50, 98, 101, 115, 117, 120, 151, 167, 175, 192, 200]
BLANK = [19, 21, 23, 69, 74, 76, 77, 81, 82, 83, 84, 89, 91, 93, 96, 104, 105, 112, 119, 125, 128, 130, 131, 133, 135, 136,
         137, 138, 140, 143, 145, 146, 149, 150, 152, 153, 154, 157, 159, 161, 169, 171, 172, 174, 176, 177, 178, 179, 180,
         182, 183, 185, 186, 187, 189, 191, 193, 194, 198, 206, 213, 214]
I = pd.read_csv(AXL / "rf_sheets/index.csv")
I["label"] = I.tile.map(lambda t: "anime" if t in ANIME else "blank" if t in BLANK else "other")
I["label_source"] = "by eye from rf_sheets v1"
I.to_parquet(AXL / "out/style_labels_tune.parquet", compression="zstd", index=False)
E = pd.read_parquet(AXL / "out/eye_e2e_tune.parquet")
j = I.merge(E, on="uuid")
print(I.label.value_counts().to_string())
a = j[j.label == "anime"]
print(a[["tile", "uuid", "status", "kind", "rel", "rel_other_kind", "rel_wrong_eye", "named_materials"]].assign(uuid=a.uuid.str[:8], status=a.status.str[:40]).to_string())
