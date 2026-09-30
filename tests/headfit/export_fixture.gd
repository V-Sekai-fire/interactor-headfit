# export_fixture -- the head-fit fixture from an avatar's model file: its face mesh at rest (USDA
# points, triangles, a submesh id per triangle) and its face shapes as sparse deltas, both in metres,
# Z up and -Y forward, the frame hf_native's marks rule reads.
#
#   godot --headless --xr-mode off --path project --script ../tests/headfit/export_fixture.gd \
#     ++ --fbx=res://../models/Mire/Mire.fbx --mesh=Body --out=res://fixtures/headfit
#
# Shape names with spaces or non-ASCII characters cannot cross the `shape <name> <n>` line format;
# they are listed and counted, never dropped silently.
extends SceneTree

const AvatarRest := preload("res://util/avatar_rest.gd")
const StickFigure := preload("res://util/stick_figure.gd")

func _arg(name: String, def: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--%s=" % name):
			return a.trim_prefix("--%s=" % name)
	return def

static func _zup(p: Vector3) -> Vector3:
	return Vector3(p.x, -p.z, p.y)

func _initialize() -> void:
	var fbx := ProjectSettings.globalize_path(_arg("fbx", "res://../models/Mire/Mire.fbx"))
	var want := _arg("mesh", "Body")
	var out := ProjectSettings.globalize_path(_arg("out", "res://fixtures/headfit"))
	var a: Dictionary = AvatarRest.export_fbx(fbx, fbx + ".meta")
	var names: PackedStringArray = a.names.split("\n", false)
	var rest: Array = []
	for i in names.size():
		rest.append(StickFigure.xf12(a.rest_local, i * 12))
	var world := StickFigure.world_pose(a.parents, rest, PackedInt32Array(), PackedFloat32Array(), -1)
	var doc := FBXDocument.new()
	var st := FBXState.new()
	doc.append_from_file(fbx, st)
	var root := doc.generate_scene(st)
	var mi: MeshInstance3D = null
	var stack: Array = [root]
	while not stack.is_empty():
		var n: Node = stack.pop_back()
		for c in n.get_children():
			stack.append(c)
		if n is MeshInstance3D and str(n.name) == want:
			mi = n
	if mi == null:
		print("FAIL no mesh ", want)
		quit(1)
		return
	var mesh: ArrayMesh = mi.mesh
	var skin: Skin = mi.skin
	var bind: Array = []
	for k in skin.get_bind_count():
		var b := skin.get_bind_bone(k)
		if b < 0:
			b = names.find(str(skin.get_bind_name(k)))
		bind.append((world[b] as Transform3D) * skin.get_bind_pose(k))
	var xyz := PackedVector3Array()
	var tri := PackedInt32Array()
	var sub := PackedInt32Array()
	var lin: Array = [] # per vertex, the blended linear part that carries a shape delta
	var shape_names := PackedStringArray()
	var skipped := PackedStringArray()
	var keep := PackedInt32Array()
	for i in mesh.get_blend_shape_count():
		var nm := str(mesh.get_blend_shape_name(i))
		if nm.is_empty() or nm.contains(" ") or nm.to_ascii_buffer().get_string_from_ascii() != nm or nm.begins_with("-"):
			skipped.append(nm)
		else:
			shape_names.append(nm)
			keep.append(i)
	var deltas: Array = []
	for k in keep.size():
		deltas.append(PackedVector3Array())
	for s in mesh.get_surface_count():
		var arr: Array = mesh.surface_get_arrays(s)
		var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
		var bones: PackedInt32Array = arr[Mesh.ARRAY_BONES]
		var w: PackedFloat32Array = arr[Mesh.ARRAY_WEIGHTS]
		var stride := bones.size() / v.size()
		var base := xyz.size()
		for i in v.size():
			var m := Transform3D(Basis(Vector3.ZERO, Vector3.ZERO, Vector3.ZERO), Vector3.ZERO)
			for j in stride:
				var wt: float = w[i * stride + j]
				if wt > 0.0:
					var t: Transform3D = bind[bones[i * stride + j]]
					m.basis.x += t.basis.x * wt
					m.basis.y += t.basis.y * wt
					m.basis.z += t.basis.z * wt
					m.origin += t.origin * wt
			xyz.append(_zup(m * v[i]))
			lin.append(m.basis)
		var idx: PackedInt32Array = arr[Mesh.ARRAY_INDEX]
		for t in range(0, idx.size(), 3):
			tri.append(base + idx[t])
			tri.append(base + idx[t + 2])
			tri.append(base + idx[t + 1])
			sub.append(s)
		var shapes: Array = mesh.surface_get_blend_shape_arrays(s)
		for k in keep.size():
			var sv: PackedVector3Array = (shapes[keep[k]] as Array)[Mesh.ARRAY_VERTEX]
			for i in v.size():
				deltas[k].append(_zup((lin[base + i] as Basis) * (sv[i] - v[i])))
	root.free()
	DirAccess.make_dir_recursive_absolute(out)
	var pts := PackedStringArray()
	for p in xyz:
		pts.append("(%.7f, %.7f, %.7f)" % [p.x, p.y, p.z])
	var counts := PackedStringArray()
	for t in tri.size() / 3:
		counts.append("3")
	var usda := "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\ndef Mesh \"%s\"\n{\n" % want
	usda += "    int[] faceVertexCounts = [%s]\n" % ", ".join(counts)
	usda += "    int[] faceVertexIndices = [%s]\n" % ", ".join(Array(tri).map(func(x): return str(x)))
	usda += "    point3f[] points = [%s]\n" % ", ".join(pts)
	usda += "    int[] primvars:submesh = [%s] (\n        interpolation = \"uniform\"\n    )\n}\n" % ", ".join(Array(sub).map(func(x): return str(x)))
	var f := FileAccess.open(out.path_join("avatar_body.usda"), FileAccess.WRITE)
	f.store_string(usda)
	f.close()
	var sh := PackedStringArray(["# shape <name> <n>, then <vertex> dx dy dz in metres (Z up) -- tests/headfit/export_fixture.gd"])
	for k in keep.size():
		var rows := PackedStringArray()
		var d: PackedVector3Array = deltas[k]
		for i in d.size():
			if d[i].length() > 1e-7:
				rows.append("%d %.7f %.7f %.7f" % [i, d[i].x, d[i].y, d[i].z])
		sh.append("shape %s %d" % [shape_names[k], rows.size()])
		sh.append_array(rows)
	f = FileAccess.open(out.path_join("avatar_shapes.txt"), FileAccess.WRITE)
	f.store_string("\n".join(sh) + "\n")
	f.close()
	print("exported %s: %d vertices, %d triangles, %d surfaces, %d shapes; %d names not exportable: %s" % [want,
			xyz.size(), tri.size() / 3, mesh.get_surface_count(), keep.size(), skipped.size(), ", ".join(skipped)])
	quit(0)
