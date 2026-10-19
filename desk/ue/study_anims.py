"""Desk probe (not committed): Mire's expression clips (Unity .anim) and three-vrm-girl's expressions.

    python3 study_anims.py
"""
import json, pathlib, re, struct, collections

A = pathlib.Path.home() / "Documents/unity-fabric/Assets/CafeMohka/Animations/Mire"


def parse_anim(p):
    t = p.read_text(errors="replace")
    name = re.search(r"m_Name: (\S+)", t).group(1)
    stop = re.search(r"m_StopTime: ([\d.]+)", t)
    loop = re.search(r"m_LoopTime: (\d)", t)
    curves = []
    for block in t.split("  - curve:")[1:]:
        attr = re.search(r"attribute: (\S+)", block)
        path = re.search(r"path: (.*)", block)
        keys = [(float(a), float(b)) for a, b in re.findall(r"time: ([-\d.e]+)\n\s+value: ([-\d.e]+)", block)]
        if attr:
            curves.append((path.group(1).strip() if path else "", attr.group(1), keys))
    return name, float(stop.group(1)) if stop else None, loop.group(1) if loop else "?", curves


print("== Mire clips")
for p in sorted(A.glob("*.[Aa]nim")):
    name, stop, loop, curves = parse_anim(p)
    moving = [(pa, at, k) for pa, at, k in curves if any(v != 0 for _, v in k)]
    print(f"{name}: length {stop} s, loop {loop}, {len(curves)} curves, {len(moving)} nonzero")
    for pa, at, k in moving:
        ks = " ".join(f"{tt:.3f}:{v:g}" for tt, v in k)
        print(f"    {pa}/{at.replace('blendShape.', '')}  {ks}")

ctl = pathlib.Path.home() / "Documents/unity-fabric/Assets/CafeMohka/Avatars/Mire/VRC/Mire.VRC.Fx.controller"
t = ctl.read_text(errors="replace")
print("== Mire FX controller")
print("layers:", re.findall(r"m_Name: (\S.*)\n\s+m_StateMachine", t))
print("parameters:", re.findall(r"m_Name: (\S+)\n\s+m_Type: (\d)", t)[:60])
# motions referenced by GUID -> anim file
guid = {}
for p in A.glob("*.meta"):
    g = re.search(r"guid: (\w+)", p.read_text())
    if g:
        guid[g.group(1)] = p.name.replace(".meta", "")
states = re.findall(r"AnimatorState:\n(?:.*\n)*?\s+m_Name: (.*)\n(?:.*\n)*?\s+m_Motion: \{fileID: (-?\d+)(?:, guid: (\w+))?", t)
for s, fid, g in states:
    print(f"  state {s.strip():28s} motion {guid.get(g, g or fid)}")
trans = re.findall(r"m_ConditionMode: (\d+)\n\s+m_ConditionEvent: (\S+)\n\s+m_EventTreshold: ([-\d.]+)", t)
print("transition conditions:", collections.Counter((e, m, th) for m, e, th in trans).most_common(40))

print("== three-vrm-girl expressions")
v = pathlib.Path.home() / "Desktop/smoke-water/3-interactor/test-godot-vrm-motion-checker/fire_motion_check/art/three_girl/three-vrm-girl.vrm"
b = v.read_bytes(); jl = struct.unpack_from("<I", b, 12)[0]; J = json.loads(b[20:20 + jl])
print("glTF animations:", len(J.get("animations", [])))
names = {}
for mi, m in enumerate(J["meshes"]):
    tn = m.get("extras", {}).get("targetNames") or (m["primitives"][0].get("extras", {}).get("targetNames") if m["primitives"] else None)
    names[mi] = tn
for g in J["extensions"]["VRM"]["blendShapeMaster"]["blendShapeGroups"]:
    binds = ", ".join(f"{(names.get(bd['mesh']) or ['?'] * 999)[bd['index']]}={bd['weight']:g}" for bd in g["binds"])
    mats = ", ".join(f"{mv['materialName']}.{mv['propertyName']}" for mv in g.get("materialValues", []))
    print(f"  {g['presetName']:10s} {g['name']:14s} binary={g.get('isBinary')}  {binds}  {('materials: ' + mats) if mats else ''}")
print("face targets:", [n for n in (names.get(0) or []) if n])
