"""v5f + open skin edges above the eye anchored."""
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.EYE_FOLLOW = True
p5.ANCHOR = 0.1
close = p5.close
