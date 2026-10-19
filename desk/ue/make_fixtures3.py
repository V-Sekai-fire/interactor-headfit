"""Native fixtures for the named-eyes rule: reuse fx2/ geometry, write proto_<kind>_left.txt from probe_named."""
import pathlib, sys
import numpy as np
sys.argv = sys.argv[:1]
import case2, case_usd, probe_close2 as pc, probe_close3h as p3, probe_named as pn
PL = (0.4, 1.0, 0.0, 0.5); SM = (0.5, 1.0, 0.2, 0.5)
SETS = {"three-vrm-girl": (pc.GIRL, "Fcl_EYE_Close_L"), "Twist": (p3.TWIST, "Fcl_EYE_Close_L"), "twintail": ("stages/twintail.vrm", "Fcl_EYE_Close_L"),
        "AliciaSolid": (pc.ALICIA, "eye_winkL2"), "Kimono": (p3.KIMONO, "eye_blink"), "Mire": (None, "face_smile"), "Maro": ("usd", "face_smile")}
for av, (path, t) in SETS.items():
    c = pc.case("Mire", None, t) if path is None else case_usd.case("Maro", "stages/marocchino.usda", "Maro", t) if path == "usd" else case2.case(av, path, t)
    if path is None or path == "usd":
        to = lambda A: A  # noqa: E731
    elif c["left_sign"] < 0:
        to = lambda A: np.stack([-A[:, 0], A[:, 2], A[:, 1]], 1)  # noqa: E731
    else:
        to = lambda A: np.stack([A[:, 0], -A[:, 2], A[:, 1]], 1)  # noqa: E731
    for kind, g in (("plain", PL), ("smile", SM)):
        out = pn.close(dict(c), kind, *g)
        f = pathlib.Path("fx2") / av / ("proto3_%s_left.txt" % kind)
        if out is None:
            f.write_text("none\n")
        else:
            np.savetxt(f, to(out), fmt="%.9g")
