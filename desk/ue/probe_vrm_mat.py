import sys, numpy as np
exec(open("probe_vrm_blink.py").read().split("for f in sys.argv")[0])
for path in sys.argv[1:]:
    J, acc = load(path); ver, side, binds = blink_binds(J)
    print("==", path.split("/")[-1], ver, side, "binds", [(J["meshes"][m]["name"], J["meshes"][m].get("extras", {}).get("targetNames", [None]*999)[i] if J["meshes"][m].get("extras") else i, round(w, 2)) for m, i, w in binds])
    for mesh, ti, w in binds:
        for prim in J["meshes"][mesh]["primitives"]:
            if ti >= len(prim.get("targets", [])):
                continue
            pos = acc(prim["attributes"]["POSITION"]); d = acc(prim["targets"][ti]["POSITION"]) * w
            used = np.zeros(len(pos), bool)
            if "indices" in prim:
                used[np.unique(acc(prim["indices"]).ravel().astype(np.int64))] = True
            else:
                used[:] = True
            mv = (np.linalg.norm(d, axis=1) > 1e-5) & used
            mat = J["materials"][prim["material"]]["name"] if "material" in prim else "?"
            if mv.any():
                dd = d[mv]
                print("  %-40s moved %5d of %6d used  mean d (x %+.2f y %+.2f z %+.2f) mm  max %.1f mm" % (mat[:40], mv.sum(), used.sum(), *(dd.mean(0) * 1e3), np.linalg.norm(dd, axis=1).max() * 1e3))
            else:
                print("  %-40s moved     0 of %6d used" % (mat[:40], used.sum()))
