import numpy as np, sys, itertools, json
sys.argv=sys.argv[:1]
import probe_close4h as p4, probe_close3h as p3, probe_close2 as pc, case2, occlude, probe_lid
p3.HULL="lo"; p4.FOLLOW=True; p4.Q=0.3
def conv(c):
    if c["up"]==2: return c, (lambda A:A)
    f=(lambda A: np.stack([-A[:,0],A[:,2],A[:,1]],1)) if c["left_sign"]<0 else (lambda A: np.stack([A[:,0],-A[:,2],A[:,1]],1))
    return dict(c,P=f(c["P"]),ref=f(c["ref"]),left_sign=1.0), f
REFS={"plain":[("three-vrm-girl",pc.GIRL,"Fcl_EYE_Close_L"),("Twist",p3.TWIST,"Fcl_EYE_Close_L"),("AliciaSolid",pc.ALICIA,"eye_winkL2"),("Kimono",p3.KIMONO,"eye_blink")],
      "smile":[("three-vrm-girl",pc.GIRL,"Fcl_EYE_Joy_L"),("Twist",p3.TWIST,"Fcl_EYE_Joy_L"),("Mire",None,"face_smile")]}
C={}
for kind,refs in REFS.items():
    for av,p,t in refs:
        c=pc.case("Mire",None,t) if p is None else case2.case(av,p,t); r=p3.eye_roles(c); cc,f=conv(c); C[(kind,av)]=(c,r,cc,f)
mire=pc.case("Mire",None,"face_smile"); mr=p3.eye_roles(mire)
def ev(kind,g):
    rows=[]
    for (k,av),(c,r,cc,f) in C.items():
        if k!=kind: continue
        gen=p4.close(c,kind,*g)
        rows.append((av,probe_lid.score(gen,c["ref"],c["sel"])[0],occlude.hidden(cc,r,f(gen))[0] if (r=="iris").any() else 1.0))
    if kind=="plain":
        rows.append(("Mire(no ref)",float("nan"),occlude.hidden(mire,mr,p4.close(mire,"plain",*g))[0]))
    return rows
kind=sys.argv[1] if len(sys.argv)>1 else "plain"
G={"plain":itertools.product((0.05,0.1,0.2,0.3),(0.6,),(0.0,0.1,0.2,0.4),(0.5,)),"smile":itertools.product((0.3,0.4),(0.6,),(0.4,),(0.25,))}
for g in G[kind]:
    rows=ev(kind,g)
    print(g," | ".join("%s %.3f/%.2f"%r for r in rows),flush=True)
