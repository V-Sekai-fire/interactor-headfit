"""Desk tool (not committed): hf_native fixtures for the anime eye-close test.

Writes fx/<avatar>/avatar_body.usda, avatar_shapes.txt and avatar_roles.txt in the fit's frame
(Z up, -Y forward, the avatar's left on +X, metres), and the Python prototype's plain close for
parity (fx/<avatar>/proto_plain_left.txt).

    python3 make_fixtures.py
"""
import json, os, pathlib, sys
import numpy as np

UE = pathlib.Path(__file__).resolve().parent
os.chdir(UE)
sys.path.insert(0, str(UE))
import probe_close2 as pc  # noqa: E402

ROLE = {"skin": 0, "eye": 1, "lash": 2, "iris": 3, "brow": 4}
P = json.load(open("close_params.json"))["plain"]["params"]


def write(dirn, P_fit, tris, roles, shapes, proto):
    d = UE / "fx" / dirn
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "avatar_body.usda", "w") as f:
        f.write('#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Mesh "Body"\n{\n')
        f.write("    int[] faceVertexCounts = [" + ", ".join(["3"] * len(tris)) + "]\n")
        f.write("    int[] faceVertexIndices = [" + ", ".join(str(int(v)) for v in tris.reshape(-1)) + "]\n")
        f.write("    point3f[] points = [" + ", ".join("(%.9g, %.9g, %.9g)" % tuple(p) for p in P_fit) + "]\n")
        f.write("    int[] primvars:submesh = [" + ", ".join(["0"] * len(tris)) + "] (\n        interpolation = \"uniform\"\n    )\n}\n")
    with open(d / "avatar_shapes.txt", "w") as f:
        for name, D in shapes.items():
            nz = np.where(np.linalg.norm(D, axis=1) > 0)[0]
            f.write(f"shape {name} {len(nz)}\n")
            for i in nz:
                f.write("%d %.9g %.9g %.9g\n" % (i, *D[i]))
    (d / "avatar_roles.txt").write_text("\n".join(str(int(r)) for r in roles) + "\n")
    np.savetxt(d / "proto_plain_left.txt", proto, fmt="%.9g")
    print("wrote", d, len(P_fit), "vertices", len(tris), "triangles", {k: int((roles == v).sum()) for k, v in ROLE.items()})


def vrm0_to_fit(A):  # Y up, facing -Z, left -X  ->  Z up, facing -Y, left +X (a proper rotation)
    return np.stack([-A[:, 0], A[:, 2], A[:, 1]], 1)


for av, path, ref in (("three-vrm-girl", pc.GIRL, "Fcl_EYE_Close_L"), ("AliciaSolid", pc.ALICIA, "eye_winkL2")):
    c = pc.case(av, path, ref)
    proto = pc.close(c, "plain", *P)
    roles = np.array([ROLE[r] for r in c["roles"]])
    write(av, vrm0_to_fit(c["P"]), c["tris"], roles, {ref: vrm0_to_fit(c["ref"])}, vrm0_to_fit(proto))

c = pc.case("Mire", None, "face_smile")
proto = pc.close(c, "plain", *P)
roles = np.array([ROLE[r] for r in c["roles"]])
write("Mire", c["P"], c["tris"], roles, {"face_smile": c["ref"]}, proto)
