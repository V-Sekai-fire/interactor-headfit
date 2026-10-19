"""v5f + weld radius 0.001 m."""
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.EYE_FOLLOW = True
p5.WELD = 0.001
close = p5.close
