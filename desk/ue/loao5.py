import sys, itertools, numpy as np
sys.argv=["x"]
import case_usd, case2, probe_close2 as pc, probe_close3h as p3, probe_close4h as p4, probe_lid
p3.HULL="lo"; p3.PARTS="widest"
L={"three-vrm-girl":lambda t:case2.case("g",pc.GIRL,t),"Twist":lambda t:case2.case("t",p3.TWIST,t),"twintail":lambda t:case2.case("tt","stages/twintail.vrm",t),
   "AliciaSolid":lambda t:case2.case("a",pc.ALICIA,t),"Kimono":lambda t:case2.case("k",p3.KIMONO,t),"Mire":lambda t:pc.case("Mire",None,t),"Maro":lambda t:case_usd.case("Maro","stages/marocchino.usda","Maro",t)}
REF={"plain":[("three-vrm-girl","Fcl_EYE_Close_L"),("Twist","Fcl_EYE_Close_L"),("twintail","Fcl_EYE_Close_L"),("AliciaSolid","eye_winkL2"),("Kimono","eye_blink")],
     "smile":[("three-vrm-girl","Fcl_EYE_Joy_L"),("Twist","Fcl_EYE_Joy_L"),("twintail","Fcl_EYE_Joy_L"),("Mire","face_smile"),("Maro","face_smile")]}
C={(k,a):L[a](t) for k in REF for a,t in REF[k]}
for c in C.values(): p3.eye_roles(c)
G={"plain":[(k,0.6,0.0,0.5,q) for k in (0.2,0.3,0.4) for q in (0.3,)],
   "smile":[(k,0.6,b,0.25,q) for k in (0.3,0.4,0.5) for b in (0.2,0.4,0.8) for q in (0.15,0.3,0.5)]}
cache={}
def rel(kind,a,g):
    if (kind,a,g) not in cache:
        p4.Q=g[4]; c=C[(kind,a)]; cache[(kind,a,g)]=probe_lid.score(p4.close(c,kind,*g[:4]),c["ref"],c["sel"])[0]
    return cache[(kind,a,g)]
best={}
for kind in REF:
    av=[a for a,_ in REF[kind]]
    print("==",kind)
    for h in av:
        tr=[a for a in av if a!=h]; g=min(G[kind],key=lambda g:max(rel(kind,a,g) for a in tr))
        print("  HELD OUT %-15s %.3f  params %s"%(h,rel(kind,h,g),g),flush=True)
    g=min(G[kind],key=lambda g:max(rel(kind,a,g) for a in av)); best[kind]=g
    print("  all: %s  %s"%(g," ".join("%s %.3f"%(a,rel(kind,a,g)) for a in av)))
for kind,other in (("plain","smile"),("smile","plain")):
    for a,_ in REF[other]:
        p4.Q=best[kind][4]; c=C[(other,a)]
        print("control %s rule on %s %-15s %.3f (own rule %.3f)"%(kind,other,a,probe_lid.score(p4.close(c,kind,*best[kind][:4]),c["ref"],c["sel"])[0],rel(other,a,best[other])))
