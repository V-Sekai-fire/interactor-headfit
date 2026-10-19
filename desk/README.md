# Anime eye close: desk probes (snapshot)

Desk scripts from `~/Desktop/stk-work/ue` and `~/Desktop/stk-work/axl`, saved so the work survives
the desk. They are not built or tested by CI. Run them from those directories (paths in them are
the desk's).

- Current rule: `ue/probe_close5.py` with the switches in `ue/probe_close5c.py` (seam copies
  merged before the close, upper lash slid per column, eye white still).
- Write closes into a VRM copy and render the textured, alpha-cutout sheet:
  `pixi run --manifest-path ../axl/pixi.toml python gen_sheet.py probe_close5c girl twist alicia`
- Hole check (magenta background, counts see-through pixels): `holes.sh <module> <avatar> <fwd>`.
- Eye-bone finder: `axl/bones_probe.py`; RF-DETR finder (did not work, too little anime data):
  `axl/render_face.py`, `detect_eyes.py`, `eye_finder.py`.

## Open
- three-vrm-girl and Twist pass the hole check (no new holes) but have not been looked at by eye
  since the connected change; Twist's lash nearly vanished before it.
- AliciaSolid: reframe the camera (the hole count sees background around her head at rest).
- twintail: closes open holes at the top of the eye opening (blink 950 px, smile 1816, own 178
  and 208). Next idea: her own blink may move the lid forward; compare first.
- Main ships no anime closes (`kAnimeClosesShip = false`, PR #11). Flip it only after the sheets
  pass by eye on every avatar. The seam-merge step must be ported to the C++ close.
