"""Write the orbit-video inputs: each avatar as a GLB whose face mesh carries two extra morph
targets at the end, `own_blink` (the avatar's own left blink, copied) and `generated_blink`, plus
a sidecar JSON with the head's centre and radius and the forward axis.

    python3 make_glbs.py <out dir>
"""
import json, math, pathlib, re, struct, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import os
os.chdir(HERE.parent)
import probe_lid  # noqa: E402

PARAMS = (0.3, 0.6, 0.25, 0.0)
S = pathlib.Path.home() / "Desktop/smoke-water"
VRMS = {
    "three-vrm-girl": S / "3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm",
    "AliciaSolid": S / "4-entities/godot-vrm/vrm_samples/AliciaSolid_vrm-0.51.vrm",
}


def read_glb(path):
    b = open(path, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    J = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl = struct.unpack_from("<I", b, off)[0]
    return J, bytearray(b[off + 8: off + 8 + bl])


def write_glb(path, J, B):
    while len(B) % 4:
        B += b"\0"
    J["buffers"][0]["byteLength"] = len(B)
    js = json.dumps(J, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(B))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(B), 0x004E4942) + bytes(B)
    open(path, "wb").write(out)


def add_vec3(J, B, arr):
    while len(B) % 4:
        B += b"\0"
    a = np.ascontiguousarray(arr, np.float32)
    J["bufferViews"].append({"buffer": 0, "byteOffset": len(B), "byteLength": a.nbytes})
    B += a.tobytes()
    J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5126, "count": len(a),
                           "type": "VEC3", "min": a.min(0).tolist(), "max": a.max(0).tolist()})
    return len(J["accessors"]) - 1


import states as ST  # noqa: E402
TARGETS = ["own plain", "generated plain", "own smile", "generated smile"]


def vrm(name, out):
    path = ST.REFS[name][0]
    J, B = read_glb(path)
    c, st = ST.states(name)
    ver, side, binds = probe_lid.blink_binds(J)
    mi_ = c["mesh_index"]
    mesh = J["meshes"][mi_]
    acc_ids = [add_vec3(J, B, st[t] if st[t] is not None else np.zeros_like(c["P"])) for t in TARGETS]
    n0 = len(mesh["primitives"][0].get("targets", []))
    for pr in mesh["primitives"]:
        pr.setdefault("targets", [])
        pr["targets"] += [{"POSITION": a} for a in acc_ids]
    names = mesh.setdefault("extras", {}).setdefault("targetNames", [f"t{k}" for k in range(n0)])
    names += [t.replace(" ", "_") for t in TARGETS]
    if "weights" in mesh:
        mesh["weights"] += [0.0] * len(TARGETS)
    J.pop("extensionsRequired", None)
    write_glb(out / f"{name}.glb", J, B)
    return {"glb": f"{name}.glb", "forward_z": -1.0 if ver == "0.x" else 1.0,
            "have": [t for t in TARGETS if st[t] is not None]}


def mire(out):
    D = HERE.parent / "dress-on/project/fixtures/headfit"
    c, st = ST.states("Mire")
    u = open(D / "avatar_body.usda").read()
    idx = np.array([int(x) for x in re.search(r"int\[\] faceVertexIndices\s*=\s*\[(.*?)\]", u, re.S).group(1).split(",")]).reshape(-1, 3)
    toy = lambda P: np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)  # noqa: E731  Z up, -Y fwd -> Y up, +Z fwd
    eye_v = np.where(c["roles"] == "eye")[0]
    tri_eye = np.isin(idx, eye_v).all(1)
    J = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 0}], "bufferViews": [], "accessors": [],
         "materials": [{"name": "skin", "pbrMetallicRoughness": {"baseColorFactor": [0.75, 0.62, 0.55, 1], "metallicFactor": 0, "roughnessFactor": 1}, "doubleSided": True},
                       {"name": "eye", "pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.25, 0.6, 1], "metallicFactor": 0, "roughnessFactor": 1}, "doubleSided": True}],
         "meshes": [], "nodes": [{"mesh": 0, "name": "Body"}], "scenes": [{"nodes": [0]}], "scene": 0}
    B = bytearray()
    a_pos = add_vec3(J, B, toy(c["P"]))
    acc_ids = [add_vec3(J, B, toy(st[t] if st[t] is not None else np.zeros_like(c["P"]))) for t in TARGETS]
    prims = []
    for m, sel in ((0, ~tri_eye), (1, tri_eye)):
        t = np.ascontiguousarray(idx[sel].reshape(-1), np.uint32)
        while len(B) % 4:
            B += b"\0"
        J["bufferViews"].append({"buffer": 0, "byteOffset": len(B), "byteLength": t.nbytes})
        B += t.tobytes()
        J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5125, "count": len(t), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": a_pos}, "indices": len(J["accessors"]) - 1, "material": m,
                      "targets": [{"POSITION": a} for a in acc_ids]})
    J["meshes"].append({"name": "Body", "primitives": prims, "weights": [0] * len(TARGETS),
                        "extras": {"targetNames": [t.replace(" ", "_") for t in TARGETS]}})
    write_glb(out / "Mire.glb", J, B)
    return {"glb": "Mire.glb", "forward_z": 1.0, "have": [t for t in TARGETS if st[t] is not None]}


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    side = {}
    for n in VRMS:
        side[n] = vrm(n, out)
    side["Mire"] = mire(out)
    json.dump(side, open(out / "avatars.json", "w"), indent=1)
    print(json.dumps(side, indent=1))
