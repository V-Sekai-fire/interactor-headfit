"""Desk tool (not committed): hf_native fixtures (fit frame: Z up, -Y forward, left +X, metres) for
every eye-close reference, with the Python v3 rule's left close for parity:
fx2/<avatar>/{avatar_body.usda, avatar_shapes.txt, avatar_roles.txt, proto_<kind>_left.txt}."""
import pathlib, sys
import numpy as np
sys.argv = sys.argv[:1]
import case_usd, case2, probe_close2 as pc, probe_close3 as p3, probe_close4h as p4
p4.p3.HULL = "lo"
p4.p3.PARTS = "widest"

PL = (0.4, 0.6, 0.0, 0.5, 0.0)
SM = (0.5, 0.6, 0.2, 0.25, 0.0)
ROLE = {"skin": 0, "eye": 1, "lash": 2, "iris": 3, "brow": 4}
SETS = {"three-vrm-girl": (pc.GIRL, ["Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"]), "Twist": (p3.TWIST, ["Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"]),
        "AliciaSolid": (pc.ALICIA, ["eye_winkL2"]), "Kimono": (p3.KIMONO, ["eye_blink"]), "Mire": (None, ["face_smile"]), "twintail": ("stages/twintail.vrm", ["Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"]), "Maro": ("usd", ["face_smile"])}
UE = pathlib.Path(".").resolve()


def write(dirn, P_fit, tris, roles, shapes, proto):
    d = UE / "fx2" / dirn
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


for av, (path, refs) in SETS.items():
    cs = [pc.case("Mire", None, t) if path is None else case_usd.case("Maro", "stages/marocchino.usda", "Maro", t) if path == "usd" else case2.case(av, path, t) for t in refs]
    c = cs[0]
    if path is None or path == "usd":
        to = lambda A: A  # noqa: E731  Mire is already in the fit frame
    elif c["left_sign"] < 0:
        to = lambda A: np.stack([-A[:, 0], A[:, 2], A[:, 1]], 1)  # noqa: E731  VRM 0.x
    else:
        to = lambda A: np.stack([A[:, 0], -A[:, 2], A[:, 1]], 1)  # noqa: E731  VRM 1.0
    roles = np.array([ROLE[r] for r in c["roles"]])
    shapes = {t: to(x["ref"]) for t, x in zip(refs, cs)}
    d = UE / "fx2" / av
    write(av, to(c["P"]), c["tris"], roles, shapes, (setattr(p4, "Q", 0.3), to(p4.close(c, "plain", *PL[:4])))[1])
    np.savetxt(d / "proto_smile_left.txt", (setattr(p4, "Q", 0.15), to(p4.close(c, "smile", *SM[:4])))[1], fmt="%.9g")
    (d / "kinds.txt").write_text("\n".join("%s %s" % (t, pc.kind_of(x)[0]) for t, x in zip(refs, cs)) + "\n")
