"""v5 without the skin-follow band (lash rigid only)."""
import probe_close5 as p5
p5.FOLLOW = False
close = p5.close
