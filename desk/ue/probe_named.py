"""Named-eyes-only close (headfit dev/anime-named-eyes): v4 without the part guessing or lash slide.
An avatar whose eye materials name no lash or iris gets no close. Leave-one-avatar-out over the
named local avatars, then the frozen Articulation-XL check lives in axl/score.py."""
import itertools
import sys

import numpy as np

sys.argv = sys.argv[:1]
import case2  # noqa: E402
import probe_close2 as pc  # noqa: E402
import probe_close3h as p3  # noqa: E402
import probe_lid  # noqa: E402

p3.HULL = "lo"


def named(c):
    return bool(np.isin(c["roles"], ["lash", "iris"]).any())


def close(c, kind, k, above, below, margin):
    if not named(c):
        return None
    c["roles2"] = c["roles"]  # no part guessing
    p3.CAP = 0.0 if kind == "plain" else None
    return p3.close(c, kind, k, above, below, margin)


REFS = {"plain": [("three-vrm-girl", pc.GIRL, "Fcl_EYE_Close_L"), ("Twist", p3.TWIST, "Fcl_EYE_Close_L"),
                  ("twintail", "stages/twintail.vrm", "Fcl_EYE_Close_L"), ("AliciaSolid", pc.ALICIA, "eye_winkL2"),
                  ("Kimono", p3.KIMONO, "eye_blink")],
        "smile": [("three-vrm-girl", pc.GIRL, "Fcl_EYE_Joy_L"), ("Twist", p3.TWIST, "Fcl_EYE_Joy_L"),
                  ("twintail", "stages/twintail.vrm", "Fcl_EYE_Joy_L")]}
GRID = {"plain": list(itertools.product((0.2, 0.3, 0.4), (0.3, 0.6, 1.0), (0.0,), (0.25, 0.5))),
        "smile": list(itertools.product((0.3, 0.4, 0.5), (0.3, 0.6, 1.0), (0.2, 0.4, 0.8), (0.25, 0.5)))}


def main():
    C = {}
    for kind, refs in REFS.items():
        for av, path, t in refs:
            c = case2.case(av, path, t)
            print("%-15s %-16s named %s roles %s" % (av, t, named(c), {r: int((c["roles"] == r).sum()) for r in sorted(set(c["roles"]))}))
            if named(c):
                C[(kind, av)] = c
    rel = lambda kind, av, g: probe_lid.score(close(C[(kind, av)], kind, *g), C[(kind, av)]["ref"], C[(kind, av)]["sel"])[0]  # noqa: E731
    best = {}
    for kind in REFS:
        av = [a for k, a in C if k == kind]
        print("==", kind)
        for h in av:
            tr = [a for a in av if a != h]
            g = min(GRID[kind], key=lambda g: max(rel(kind, a, g) for a in tr))
            print("  HELD OUT %-15s %.3f params %s" % (h, rel(kind, h, g), g))
        best[kind] = min(GRID[kind], key=lambda g: max(rel(kind, a, g) for a in av))
        print("  all %s: %s" % (best[kind], " ".join("%s %.3f" % (a, rel(kind, a, best[kind])) for a in av)))
    for kind, other in (("plain", "smile"), ("smile", "plain")):
        for k, a in C:
            if k == other:
                c = C[(k, a)]
                print("control %s rule on %s %-15s %.3f (own %.3f)" % (kind, other, a, probe_lid.score(close(c, kind, *best[kind]), c["ref"], c["sel"])[0], rel(other, a, best[other])))
    return best


if __name__ == "__main__":
    main()
