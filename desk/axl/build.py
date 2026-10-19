"""Build the anime eye-close reference set from chibifire/datasource-articulation-xl2-parquet's
anime/avatar rows: ETNF parquet tables in out/, keyed by the Articulation-XL uuid.

  assets.parquet          (uuid) split, url, file_type, http status, bytes, sha256, glb status
  materials.parquet       (uuid, mesh_ordinal, primitive_ordinal) material name, role, role rule
  morph_targets.parquet   (uuid, mesh_ordinal, target_ordinal) name
  eye_refs.parquet        (uuid, mesh_ordinal, target_ordinal) side, name rule (derived claim)
  sources.parquet         the parquet mirror revision and the fetch log

The split is the mirror's own (train -> tune, test -> frozen); FBX goes to GLB with Godot's ufbx
importer (conv/fbx2glb.gd) first."""
import json
import pathlib
import re
import struct
import subprocess
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
GLB = ROOT / "glb"
OUT = ROOT / "out"
GODOT = pathlib.Path.home() / "Desktop/smoke-water/5-repository/godot-double/godot.macos.editor.double.arm64"
ROLE_RULE = "probe_lid.role_of v1 (material name, lower-cased; iris > lash > brow > eye)"
EYE_RULE = "eye-close names v1"
# Material name -> face role, the probe's rule (ue/probe_lid.py role_of), copied so this builds alone.
ROLE = [("iris", r"iris|pupil|hitomi|highlight|hilight"), ("lash", r"lash|matsuge|mastuge|eyeline|eye_line"),
        ("brow", r"brow|mayu"), ("eye", r"eye|white")]


def role_of(name):
    for r, rx in ROLE:
        if re.search(rx, name, re.I):
            return r
    return "skin"

# (pattern, side) over the target name with any "<mesh>." prefix dropped; first match wins.
EYE_NAMES = [
    (r"fcl_eye_close_l$|eye_?blink_?l(eft)?$|blink_?l(eft)?$|eyeblinkleft$|eye-?closel$|eye_?close_?l(eft)?$|wink_?l(eft)?$|eye-?winkl$|eye_winkl2?$|ak_09_eyeblinkleft$|au_43_l_eyeclosed$", "L"),
    (r"fcl_eye_close_r$|eye_?blink_?r(ight)?$|blink_?r(ight)?$|eyeblinkright$|eye-?closer$|eye_?close_?r(ight)?$|wink_?r(ight)?$|eye-?winkr$|eye_winkr2?$|ak_10_eyeblinkright$|au_43_r_eyeclosed$", "R"),
    (r"fcl_eye_joy_l$|joy_?l$|smile_?eye_?l$|eye_?smile_?l$", "L-smile"),
    (r"fcl_eye_joy_r$|joy_?r$|smile_?eye_?r$|eye_?smile_?r$", "R-smile"),
    (r"fcl_eye_close$|^blink$|eye_?blink$|eyes_?closed$|face_blink$|vrc\.blink$|まばたき$", "both"),
    (r"fcl_eye_joy$|face_smile$|eye_?smile$|笑い$", "both-smile"),
]


def glb_json(path):
    b = path.read_bytes()
    if b[:4] != b"glTF":
        return None
    jl = struct.unpack_from("<I", b, 12)[0]
    return json.loads(b[20:20 + jl])


def target_names(m):
    return (m.get("extras") or {}).get("targetNames") or ((m.get("primitives") or [{}])[0].get("extras") or {}).get("targetNames") or []


def eye_side(name):
    n = name.split(".")[-1].lower()
    for pat, side in EYE_NAMES:
        if re.search(pat, n):
            return side
    return None


def convert_fbx(assets):
    GLB.mkdir(exist_ok=True)
    todo = [a for a in assets if a["file_type"] in ("fbx",) and a.get("sha256") and not (GLB / (a["uuid"] + ".glb")).exists()]
    status = {}
    for i in range(0, len(todo), 50):
        lst = ROOT / "conv" / "list.txt"
        lst.write_text("".join("%s\t%s\n" % (RAW / "files" / (a["uuid"] + ".fbx"), GLB / (a["uuid"] + ".glb")) for a in todo[i:i + 50]))
        try:
            r = subprocess.run([str(GODOT), "--path", str(ROOT / "conv"), "--resolution", "64x64", "--script", "fbx2glb.gd", "--", str(lst)],
                               capture_output=True, text=True, timeout=900)
            out = r.stdout
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        for line in out.splitlines():
            if line.startswith("CONV "):
                _, st, src = line.split(" ", 2)
                status[pathlib.Path(src).stem] = st
        print("converted %d / %d" % (min(i + 50, len(todo)), len(todo)), flush=True)
    return status


def main():
    meta = pd.read_csv(ROOT / "anime_rows.csv")
    split = pd.concat([pd.read_parquet(ROOT / s / "models.parquet", columns=["split", "uuid"]) for s in ("train", "test", "diverse_pose_train", "diverse_pose_test")])
    meta = meta.merge(split, on="uuid", how="left")
    req = {}
    for line in (RAW / "requests.jsonl").open():
        r = json.loads(line)
        req[r["uuid"]] = r
    assets = []
    for r in meta.itertuples():
        q = req.get(r.uuid, {})
        assets.append(dict(uuid=r.uuid, split=r.split, url=q.get("url"), file_type=r.file_type, category_label=r.category_label,
                           http_status=q.get("status"), bytes=q.get("bytes"), sha256=q.get("sha256"), fetch_error=q.get("error"),
                           lfs_pointer=bool(q.get("lfs_pointer", False)), fetched_at=q.get("fetched_at")))
    conv = convert_fbx(assets)
    mats, targets, refs = [], [], []
    for a in assets:
        if not a["sha256"] or a["lfs_pointer"]:
            a["glb_status"] = "no file"
            continue
        if a["file_type"] == "glb":
            p = RAW / "files" / (a["uuid"] + ".glb")
            a["glb_status"] = "original"
        elif a["file_type"] == "fbx":
            p = GLB / (a["uuid"] + ".glb")
            a["glb_status"] = "converted" if p.exists() else "convert " + conv.get(a["uuid"], "fail")
        else:
            a["glb_status"] = "not converted (%s)" % a["file_type"]
            continue
        J = glb_json(p) if p.exists() else None
        if J is None:
            if p.exists():
                a["glb_status"] = "not a binary glTF"
            continue
        for mi, m in enumerate(J.get("meshes", [])):
            for pi, pr in enumerate(m.get("primitives", [])):
                name = J["materials"][pr["material"]].get("name", "") if "material" in pr and J.get("materials") else ""
                mats.append(dict(uuid=a["uuid"], mesh_ordinal=mi, primitive_ordinal=pi, material=name, role=role_of(name), role_rule=ROLE_RULE))
            for ti, n in enumerate(target_names(m)):
                targets.append(dict(uuid=a["uuid"], mesh_ordinal=mi, target_ordinal=ti, name=n))
                s = eye_side(n)
                if s:
                    refs.append(dict(uuid=a["uuid"], mesh_ordinal=mi, target_ordinal=ti, side=s, rule=EYE_RULE))
    OUT.mkdir(exist_ok=True)
    tabs = {"assets": (pd.DataFrame(assets), ["uuid"]), "materials": (pd.DataFrame(mats), ["uuid", "mesh_ordinal", "primitive_ordinal"]),
            "morph_targets": (pd.DataFrame(targets), ["uuid", "mesh_ordinal", "target_ordinal"]),
            "eye_refs": (pd.DataFrame(refs), ["uuid", "mesh_ordinal", "target_ordinal"])}
    src = pd.read_parquet(ROOT / "sources.parquet")
    tabs["sources"] = (pd.DataFrame([dict(source_id="articulation-xl2-parquet", repo="chibifire/datasource-articulation-xl2-parquet",
                                          upstream=src.repo.iloc[0], upstream_revision=src.revision.iloc[0], note="meta.parquet file_identifier rows matching anime/avatar"),
                                     dict(source_id="fetch-log", repo=str(RAW / "requests.jsonl"), upstream="github raw", upstream_revision="",
                                          note="one line per request; built %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))]), ["source_id"])
    for name, (df, pk) in tabs.items():
        assert not df.duplicated(pk).any(), name + " primary key"
        if name != "assets" and name != "sources":
            assert df.uuid.isin(tabs["assets"][0].uuid).all(), name + " orphan"
        df.to_parquet(OUT / (name + ".parquet"), compression="zstd", index=False)
        print("%-14s rows %6d cols %2d" % (name, len(df), df.shape[1]))
    A = tabs["assets"][0]
    print(A.glb_status.value_counts().to_string())
    R = tabs["eye_refs"][0]
    if len(R):
        R = R.merge(A[["uuid", "split"]], on="uuid")
        print("assets with an eye-close target, by split:", R.groupby("split").uuid.nunique().to_dict())
        print(R.groupby(["split", "side"]).uuid.nunique().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
