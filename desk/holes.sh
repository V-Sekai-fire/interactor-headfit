#!/bin/bash
# usage: holes.sh <module> <avatar> <fwd>
cd ~/Desktop/stk-work/ue
case $2 in girl|twist|twintail) O=Fcl_EYE_Close_L,Fcl_EYE_Close_R; S=Fcl_EYE_Joy_L,Fcl_EYE_Joy_R;; alicia) O=eye_winkL2,eye_winkR2; S=;; esac
HOLES=1 F=gen_$1/$2.vrm OWN=$O OWNS=$S OUT=/tmp/h_$1_$2.png TITLE=x FWD=$3 pixi run --manifest-path ../axl/pixi.toml python ../axl/orbit_sheet.py >/dev/null 2>&1
python3 - /tmp/h_$1_$2.png <<'P'
import sys; from PIL import Image; import numpy as np
a=np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int); n=(a.shape[0]-30)//300
out=[]
for r_ in range(n):
  t=a[30+r_*300:30+(r_+1)*300, 140+2*300:140+3*300]; m=(t[...,0]>230)&(t[...,1]<30)&(t[...,2]>230); out.append(int(m[:200].sum()))
print("hole px (upper front) by row:", out)
P
