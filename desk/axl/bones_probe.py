"""Approach B: eyes from the VRM humanoid eye bones (spec-defined, no names, no learning).
iris := face-mesh vertices whose strongest joint is leftEye/rightEye (or a child of it).
Graded on the named avatars: names used ONLY to report agreement, never by the rule."""
import sys
import numpy as np
sys.argv = ["x"]
sys.path.insert(0, "/Users/ernest.lee/Desktop/stk-work/ue"); sys.path.insert(0, "/Users/ernest.lee/Desktop/stk-work/axl")
import case2, cues, probe_close2 as pc, probe_close3h as p3, probe_lid

p3.HULL = "lo"
PARAMS = {"plain": (0.4, 1.0, 0.0, 0.5), "smile": (0.5, 1.0, 0.2, 0.5)}
REFS = [("girl", pc.GIRL, "Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"), ("Twist", p3.TWIST, "Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"),
        ("twintail", "/Users/ernest.lee/Desktop/stk-work/ue/stages/twintail.vrm", "Fcl_EYE_Close_L", "Fcl_EYE_Joy_L"),
        ("Alicia", pc.ALICIA, "eye_winkL2", None), ("Kimono", p3.KIMONO, "eye_blink", None)]


def eye_nodes(J):
    hb = []
    v0 = J.get("extensions", {}).get("VRM", {}).get("humanoid", {}).get("humanBones", [])
    hb += [b["node"] for b in v0 if b.get("bone") in ("leftEye", "rightEye")]
    v1 = J.get("extensions", {}).get("VRMC_vrm", {}).get("humanoid", {}).get("humanBones", {})
    hb += [v1[k]["node"] for k in ("leftEye", "rightEye") if k in v1]
    out, stack = set(), list(hb)
    while stack:
        n = stack.pop(); out.add(n); stack += J["nodes"][n].get("children", [])
    return hb, out


def bone_roles(path, target):
    J, acc = cues.load_glb(path)
    for mi, m in enumerate(J["meshes"]):
        hit = [k for k, n in enumerate(case2.names_of(J, m)) if n.endswith(target)]
        if hit:
            break
    _, _, _, Jt, Wt, _ = cues.mesh_world(J, acc, mi, hit[0])
    node = [i for i, nd in enumerate(J["nodes"]) if nd.get("mesh") == mi][0]
    joints = J["skins"][J["nodes"][node]["skin"]]["joints"]
    hb, eyes = eye_nodes(J)
    dom = np.array(joints)[Jt[np.arange(len(Jt)), Wt.argmax(1)]]
    R = np.where(np.isin(dom, list(eyes)) & (Wt.max(1) > 0), "iris", "skin").astype(object)
    return R, len(hb)


for name, path, plain, smile in REFS:
    for kind, t in (("plain", plain), ("smile", smile)):
        if t is None:
            continue
        try:
            c = case2.case(name, path, t)
        except FileNotFoundError:
            print(name, "missing file"); break
        named = c["roles"].copy()
        R, nb = bone_roles(path, t)
        agree = "bones %d, eye-bone verts %d, of which named iris %d / named iris %d" % (nb, (R == "iris").sum(), ((R == "iris") & (named == "iris")).sum(), (named == "iris").sum())
        p3.CAP = 0.0 if kind == "plain" else None
        res = {}
        for lab, roles in (("names", named), ("bones", R)):
            if not (roles == "iris").any() and not (roles == "lash").any():
                res[lab] = None; continue
            cc = dict(c, roles=roles, roles2=roles)
            out = p3.close(cc, kind, *PARAMS[kind])
            res[lab] = probe_lid.score(out, c["ref"], c["sel"])[0] if out is not None else None
        print("%-8s %-5s names %s  bones %s  | %s" % (name, kind, *(("%.3f" % res[k]) if res[k] is not None else "  -  " for k in ("names", "bones")), agree))
