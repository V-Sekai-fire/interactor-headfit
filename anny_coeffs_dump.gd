# The phenotype anchor rule by anny-creator's own GDScript (scripts/anny_coeffs.gd,
# AnnyTables from assets/anny_tables.json), for Gate 11's parity check: one line
# `case <k> <mac_name> <coefficient>` per macro row per case of cases.txt (six
# phenotype values a line). Run by gates/11-headfit/parity.sh in a scratch
# project holding copies of the two scripts; anny-creator's checkout is not
# touched.
#
#   godot --headless --path <scratch> --script res://anny_coeffs_dump.gd -- out=<file> [--plant]
#
# --plant mis-orders the age anchors (anny-creator's own planted control),
# which the comparison must catch.
extends SceneTree

func _initialize() -> void:
	var args := {}
	for a in OS.get_cmdline_user_args():
		var i: int = a.find("=")
		if i > 0:
			args[a.substr(0, i)] = a.substr(i + 1)
	var t := AnnyTables.new()
	t.meta = JSON.parse_string(FileAccess.get_file_as_string("res://anny_tables.json"))
	t.target_count = int(t.meta["target_count"])
	t.free_slots = PackedStringArray(t.meta["free_slots"])
	t.axis_slots = t.meta["axis_slots"]
	t.anchors = t.meta["anchors"]
	t.target_slots = []
	for entry in t.meta["targets"]:
		t.target_slots.append(entry["slots"])
	if "--plant" in OS.get_cmdline_user_args():
		t.anchors["age"] = [1.0, 0.6666, 0.3333, 0.0, -0.3333]
	var lines := PackedStringArray()
	var k := 0
	for line in FileAccess.get_file_as_string("res://cases.txt").split("\n"):
		var v := line.split(" ", false)
		if v.size() != 6 or line.begins_with("#"):
			continue
		var ph := {}
		for a in 6:
			ph[AnnyCoeffs.AXES[a]] = v[a].to_float()
		var c := AnnyCoeffs.coefficients(t, ph, {}, {})
		for r in t.target_count:
			var nm: String = str(t.meta["targets"][r].get("name", ""))
			if nm.begins_with("mac_"):
				lines.append("case %d %s %.12f" % [k, nm, c[r]]) # String % has no %g
		k += 1
	var f := FileAccess.open(args.get("out", "res://coeffs.txt"), FileAccess.WRITE)
	f.store_string("\n".join(lines) + "\n")
	f.close()
	print("anny_coeffs_dump cases %d rows %d" % [k, lines.size()])
	quit(0)
