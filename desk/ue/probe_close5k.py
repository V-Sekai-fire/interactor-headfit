import numpy as np
import probe_close5 as p5
p5.FOLLOW = False
p5.EYE_STILL = True
p5.UNIFORM = 0.5
def close(c, kind, params=None):
    out = p5.close(c, kind, params)
    out[c["roles"] != "skin"] = 0.0
    return out
