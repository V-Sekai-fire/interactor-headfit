"""Write a probe's closes into a copy of a source VRM as EyeClosedLeft/Right and
EyeClosedSmileLeft/Right on the face mesh, score the left close against the avatar's own, and
render the textured alpha-cutout sheet.
  python gen_sheet.py <probe module> <avatar> ...   (avatars: girl twist alicia twintail kimono)"""
import importlib
import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np

UE = pathlib.Path(__file__).parent
sys.path.insert(0, str(UE))
mod = sys.argv[1]
names = sys.argv[2:]
sys.argv = sys.argv[:1]
import case2  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_close3h as p3  # noqa: E402
import probe_lid  # noqa: E402

pr = importlib.import_module(mod)
AV = {"girl": (pc.GIRL, "Fcl_EYE_Close_L", "Fcl_EYE_Close_R", "Fcl_EYE_Joy_L", "Fcl_EYE_Joy_R"),
      "twist": (p3.TWIST, "Fcl_EYE_Close_L", "Fcl_EYE_Close_R", "Fcl_EYE_Joy_L", "Fcl_EYE_Joy_R"),
      "twintail": (str(UE / "stages/twintail.vrm"), "Fcl_EYE_Close_L", "Fcl_EYE_Close_R", "Fcl_EYE_Joy_L", "Fcl_EYE_Joy_R"),
      "alicia": (pc.ALICIA, "eye_winkL2", "eye_winkR2", None, None)}


def write(src, out, mesh_target, shapes):
    b = open(src, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    at = 20 + jl
    B = bytearray(b[at + 8:at + 8 + struct.unpack_from("<I", b, at)[0]])
    for mi_, m in enumerate(J["meshes"]):
        if any(n.endswith(mesh_target) for n in case2.names_of(J, m)):
            break
    tn = list(case2.names_of(J, m))
    first, order = {}, []
    for p in m["primitives"]:
        a = int(p["attributes"]["POSITION"])
        if a not in first:
            first[a] = sum(int(J["accessors"][x]["count"]) for x in order)
            order.append(a)
    for name, D in shapes.items():
        made = {}
        for a in order:
            n = int(J["accessors"][a]["count"])
            arr = np.ascontiguousarray(D[first[a]:first[a] + n], np.float32)
            while len(B) % 4:
                B += b"\0"
            J["bufferViews"].append({"buffer": 0, "byteOffset": len(B), "byteLength": arr.nbytes})
            B += arr.tobytes()
            J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5126, "count": n, "type": "VEC3",
                                   "min": arr.min(0).tolist(), "max": arr.max(0).tolist()})
            made[a] = len(J["accessors"]) - 1
        for p in m["primitives"]:
            p.setdefault("targets", []).append({"POSITION": made[int(p["attributes"]["POSITION"])]})
        tn.append(name)
        if "weights" in m:
            m["weights"].append(0.0)
    m.setdefault("extras", {})["targetNames"] = tn
    for p in m["primitives"]:
        if "extras" in p and "targetNames" in p["extras"]:
            p["extras"]["targetNames"] = tn
    while len(B) % 4:
        B += b"\0"
    J["buffers"][0]["byteLength"] = len(B)
    js = json.dumps(J, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    o = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(B)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(B), 0x004E4942) + bytes(B)
    open(out, "wb").write(o)


outdir = UE / ("gen_" + mod)
outdir.mkdir(exist_ok=True)
for av in names:
    path, pl, pr_, sl, sr = AV[av]
    c = case2.case(av, path, pl)
    shapes = {}
    for kind, tag in (("plain", "EyeClosed"), ("smile", "EyeClosedSmile")):
        L = pr.close(c, kind)
        R = pr.close(dict(c, left_sign=-c["left_sign"]), kind)
        shapes[tag + "Left"], shapes[tag + "Right"] = L, R
        for ref_t in ([pl] if kind == "plain" else ([sl] if sl else [])):
            cr = case2.case(av, path, ref_t)
            print("%-8s %-5s rel %.3f  (wrong eye %.3f)" % (av, kind, probe_lid.score(L, cr["ref"], cr["sel"])[0], probe_lid.score(R, cr["ref"], cr["sel"])[0]))
    out = outdir / (av + ".vrm")
    write(path, str(out), pl, shapes)
    env = dict(os.environ, F=str(out), OWN=pl + "," + pr_, OUT=str(outdir / (av + ".png")), TITLE="%s, %s" % (av, mod),
               FWD=str(float(c["fwd"][2])))
    if sl:
        env["OWNS"] = sl + "," + sr
    r = subprocess.run([sys.executable, str(UE.parent / "axl/orbit_sheet.py")], env=env, capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])
