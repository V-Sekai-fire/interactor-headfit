"""The states each avatar shows: rest, then own/generated for the plain and the smile close."""
import json, os, pathlib, sys
import numpy as np

UE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UE))
_cwd = os.getcwd()
os.chdir(UE)
import probe_close2 as pc  # noqa: E402
_argv = list(sys.argv)
import probe_close3 as p3  # noqa: E402
import probe_close4h as p4  # noqa: E402
p4.p3.HULL = "lo"
p4.FOLLOW = True
import os as _os
p4.Q = float(_os.environ.get("Q", "0.3"))
sys.argv[:] = _argv
os.chdir(_cwd)

P = {"plain": {"params": [0.3, 0.6, 0.0, 0.5]}, "smile": {"params": [0.4, 0.6, 0.4, 0.25]}}  # headfit kPlainClose / kSmileClose
REFS = {"three-vrm-girl": (pc.GIRL, "Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"),
        "AliciaSolid": (pc.ALICIA, "eye_winkL2", None),
        "Mire": (None, None, "face_wink_L")}
COLUMNS = ["rest", "own plain", "generated plain", "own smile", "generated smile"]


def states(avatar):
    """{column: delta array or None (no such shape)} on the avatar's case positions, plus the case."""
    path, plain, smile = REFS[avatar]
    os.chdir(UE)
    try:
        c = pc.case(avatar, path, plain or smile)
        out = {"rest": np.zeros_like(c["P"])}
        for kind, t in (("plain", plain), ("smile", smile)):
            out["own " + kind] = pc.case(avatar, path, t)["ref"] if t else None
            out["generated " + kind] = p4.close(c, kind, *P[kind]["params"])
    finally:
        os.chdir(_cwd)
    return c, out
