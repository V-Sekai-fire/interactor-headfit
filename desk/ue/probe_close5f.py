"""v5: lash rigid, no skin band, eye white follows the nearest lid vertex, welded seams."""
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.EYE_FOLLOW = True
close = p5.close
