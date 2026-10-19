"""v5: the mesh treated as connected (seam copies merged before the close), lash rigid per
column, eye white still."""
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.CONNECTED = True
close = p5.close
