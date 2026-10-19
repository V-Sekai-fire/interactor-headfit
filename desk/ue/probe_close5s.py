"""v5f + skin slits above the eye kept shut."""
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.EYE_FOLLOW = True
p5.SLITS = True
close = p5.close
