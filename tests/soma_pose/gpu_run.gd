# The SOMA-X pose kernels on the GPU: the SPIR-V that kernels/anny/gen.sh builds, run through a
# Godot RenderingDevice (Metal on macOS) in a windowed Godot, never --headless. For every sampled
# frame of each clip it runs the whole per-frame chain and writes the vertices for tests/soma_pose
# to compare with SOMA-X (soma_pose_test --gpu): the kernels as they are, the three controls
# (correctives off, twist joints skipped, one frame late) and the Kabsch twist form. It also times
# the chain: all frames of a clip in one submission, without readback.
#
#   SOMA_FIXTURE=<fixture> SOMA_SPV=<build/spv-anny> SOMA_OUT=<dir> godot --path <project> -s gpu_run.gd
extends SceneTree

const NONE_SET := []
var rd: RenderingDevice
var fx: String
var spv: String
var out: String
var meta: Dictionary
var shaders := {}
var pipes := {}

func _init() -> void:
	fx = OS.get_environment("SOMA_FIXTURE")
	spv = OS.get_environment("SOMA_SPV")
	out = OS.get_environment("SOMA_OUT")
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		print("FAIL soma gpu: no RenderingDevice")
		quit(1)
		return
	print("device: %s" % rd.get_device_name())
	meta = JSON.parse_string(FileAccess.get_file_as_string(fx + "/meta.json"))
	var rc := _run()
	quit(rc)

func _bytes(path: String) -> PackedByteArray:
	var b := FileAccess.get_file_as_bytes(path)
	if b.is_empty():
		push_error("cannot read " + path)
	return b

func _pad(b: PackedByteArray) -> PackedByteArray:
	var n := maxi(16, (b.size() + 15) / 16 * 16)
	var c := b.duplicate()
	c.resize(n)
	return c

func _sbuf(b: PackedByteArray) -> RID:
	var p := _pad(b)
	return rd.storage_buffer_create(p.size(), p)

func _zeros(n_bytes: int) -> RID:
	var b := PackedByteArray()
	b.resize(n_bytes)
	return _sbuf(b)

func _table(name: String) -> RID:
	return _sbuf(_bytes(fx + "/tables/" + name + ".bin"))

func _params(vals: Array) -> RID:
	var b := PackedByteArray()
	b.resize(16)
	for i in vals.size():
		b.encode_u32(4 * i, vals[i])
	return rd.uniform_buffer_create(16, b)

func _shader(k: String) -> void:
	var s := RDShaderSPIRV.new()
	s.set_stage_bytecode(RenderingDevice.SHADER_STAGE_COMPUTE, _bytes(spv + "/" + k + ".spv"))
	if s.compile_error_compute != "":
		push_error(s.compile_error_compute)
	var sh := rd.shader_create_from_spirv(s, k)
	shaders[k] = sh
	pipes[k] = rd.compute_pipeline_create(sh)

# A uniform set: binding 0 the params block, bindings 1.. the storage buffers in order.
func _uset(k: String, params: RID, bufs: Array) -> RID:
	var us := []
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
	u.binding = 0
	u.add_id(params)
	us.append(u)
	for i in bufs.size():
		var s := RDUniform.new()
		s.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		s.binding = i + 1
		s.add_id(bufs[i])
		us.append(s)
	return rd.uniform_set_create(us, shaders[k], 0)

func _run() -> int:
	for k in ["soma_orient", "soma_jump", "soma_twist", "soma_expand", "soma_skin", "soma_corr_l1", "soma_corr_l2", "anny_lbs"]:
		_shader(k)
	var NJ: int = meta["NJ"]
	var NT: int = meta["NT"]
	var NS: int = meta["NS"]
	var NTW: int = meta["NTW"]
	var V: int = meta["V"]
	var D: int = meta["D"]
	var K: int = meta["K"]
	var rp: int = meta["rounds_pub"]
	var rt: int = meta["rounds_tgt"]
	# Static tables.
	var orient := _table("orient")
	var orient_pt := _table("orient_pt")
	var pub_local_t := _table("pub_local_t")
	var pub_parent := _table("pub_parent")
	var corr_bind := _table("corr_bind")
	var src_axis := _table("src_axis")
	var src_sign := _table("src_sign")
	var seg_start := _table("seg_start")
	var seg_end := _table("seg_end")
	var seg_parent := _table("seg_parent")
	var seg_rev := _table("seg_rev")
	var bind_rot := _table("bind_rot")
	var align_rot := _table("align_rot")
	var twist_m := _table("twist_m")
	var twist_m0 := _zeros(4 * 3 * NTW * NJ)
	var tw_axis := _table("tw_axis")
	var tw_sign := _table("tw_sign")
	var kind := _table("tgt_kind")
	var ref := _table("tgt_ref")
	var par := _table("tgt_par")
	var base_rot := _table("tgt_base_rot")
	var tgt_local_t := _table("tgt_local_t")
	var inv_bind := _table("inv_bind")
	var w1 := _table("w1")
	var rest := _table("rest")
	var rowptr := _table("w2_rowptr")
	var col := _table("w2_col")
	var val := _table("w2_val")
	var val0 := _zeros(4 * 3 * int(meta["E"]))
	var weights := _table("weights")
	# Per-frame buffers.
	var pose := _zeros(4 * 9 * NJ)
	var transl := _zeros(16)
	var rabs := _zeros(4 * 9 * NJ)
	var acc := [_zeros(4 * 12 * NJ), _zeros(4 * 12 * NJ)]
	var anc := [_zeros(4 * NJ), _zeros(4 * NJ)]
	var feat := _zeros(4 * 6 * NJ)
	var chan := _zeros(4 * 3 * NJ)
	var tacc := [_zeros(4 * 12 * NT), _zeros(4 * 12 * NT)]
	var tanc := [_zeros(4 * NT), _zeros(4 * NT)]
	var bone := _zeros(4 * 12 * NT)
	var z := _zeros(4 * K)
	var bind := _zeros(4 * 3 * V)
	var verts := _zeros(4 * 3 * V)
	var pub_world: RID = acc[rp % 2]
	var world: RID = tacc[rt % 2]
	# Uniform sets.
	var s_orient := _uset("soma_orient", _params([NJ, int(meta["gtj"])]),
		[pose, transl, orient, orient_pt, pub_local_t, pub_parent, corr_bind, rabs, acc[0], anc[0], feat])
	var p_jp := _params([NJ])
	var s_jp := [_uset("soma_jump", p_jp, [acc[0], anc[0], acc[1], anc[1]]), _uset("soma_jump", p_jp, [acc[1], anc[1], acc[0], anc[0]])]
	var p_jt := _params([NT])
	var s_jt := [_uset("soma_jump", p_jt, [tacc[0], tanc[0], tacc[1], tanc[1]]), _uset("soma_jump", p_jt, [tacc[1], tanc[1], tacc[0], tanc[0]])]
	var tw_bufs := [rabs, pub_world, src_axis, src_sign, seg_start, seg_end, seg_parent, seg_rev, bind_rot, align_rot, chan]
	var s_twist := [_uset("soma_twist", _params([NJ, NS, 0]), tw_bufs), _uset("soma_twist", _params([NJ, NS, 1]), tw_bufs)]
	var p_ex := _params([NT, NJ, NTW])
	var s_expand := [_uset("soma_expand", p_ex, [kind, ref, par, base_rot, tgt_local_t, twist_m, chan, tw_axis, tw_sign, pub_world, tacc[0], tanc[0]]),
		_uset("soma_expand", p_ex, [kind, ref, par, base_rot, tgt_local_t, twist_m0, chan, tw_axis, tw_sign, pub_world, tacc[0], tanc[0]])]
	var s_skin := _uset("soma_skin", _params([NT]), [world, inv_bind, bone])
	var s_l1 := _uset("soma_corr_l1", _params([D, K, int(meta["use_tanh"])]), [feat, w1, z])
	var p_l2 := _params([V])
	var s_l2 := [_uset("soma_corr_l2", p_l2, [rest, rowptr, col, val, z, bind]), _uset("soma_corr_l2", p_l2, [rest, rowptr, col, val0, z, bind])]
	var s_lbs := _uset("anny_lbs", _params([V, NT]), [bind, weights, bone, verts])
	var groups := func(n: int) -> int: return (n + 63) / 64

	# One frame's chain in the open compute list cl. form: twist form; mz: twist matrix zeroed;
	# w0: correctives off.
	var chain := func(cl: int, form: int, mz: bool, w0: bool) -> void:
		var steps := [["soma_orient", s_orient, NJ]]
		for r in rp:
			steps.append(["soma_jump", s_jp[r % 2], NJ])
		steps.append(["soma_twist", s_twist[form], NJ])
		steps.append(["soma_expand", s_expand[1 if mz else 0], NT])
		for r in rt:
			steps.append(["soma_jump", s_jt[r % 2], NT])
		steps.append(["soma_skin", s_skin, NT])
		steps.append(["soma_corr_l1", s_l1, K])
		steps.append(["soma_corr_l2", s_l2[1 if w0 else 0], V])
		steps.append(["anny_lbs", s_lbs, V])
		for st in steps:
			rd.compute_list_bind_compute_pipeline(cl, pipes[st[0]])
			rd.compute_list_bind_uniform_set(cl, st[1], 0)
			rd.compute_list_dispatch(cl, groups.call(st[2]), 1, 1)
			rd.compute_list_add_barrier(cl)

	var report := ""
	for clip in meta["clips"]:
		var cd: String = fx + "/clips/" + clip + "/"
		var all_pose := _bytes(cd + "pose.bin")
		var all_transl := _bytes(cd + "transl.bin")
		var frames := _bytes(cd + "frames.bin").to_int32_array()
		var od: String = out + "/" + clip
		DirAccess.make_dir_recursive_absolute(od)
		# Form 1 (SOMA-X's half-angle twist) for the parity runs, form 0 (the Kabsch split, the
		# default) for the comparison and the timing.
		var variants := {"verts": [1, false, false, 0], "off": [1, false, true, 0], "tw": [1, true, false, 0],
			"shift": [1, false, false, 1], "kabsch": [0, false, false, 0]}
		for name in variants:
			var vv: Array = variants[name]
			var f := FileAccess.open(od + "/" + name + ".bin", FileAccess.WRITE)
			for fr in frames:
				var g: int = fr + vv[3]
				rd.buffer_update(pose, 0, 4 * 9 * NJ, all_pose.slice(4 * 9 * NJ * g, 4 * 9 * NJ * (g + 1)))
				rd.buffer_update(transl, 0, 12, all_transl.slice(12 * g, 12 * (g + 1)))
				var cl := rd.compute_list_begin()
				chain.call(cl, vv[0], vv[1], vv[2])
				rd.compute_list_end()
				rd.submit()
				rd.sync()
				f.store_buffer(rd.buffer_get_data(verts, 0, 4 * 3 * V))
			f.close()
		# Timing: every sampled frame in one submission (pose upload + chain, no readback), best of 5.
		var best := 1e30
		for rep in 5:
			var t0 := Time.get_ticks_usec()
			for fr in frames:
				rd.buffer_update(pose, 0, 4 * 9 * NJ, all_pose.slice(4 * 9 * NJ * fr, 4 * 9 * NJ * (fr + 1)))
				rd.buffer_update(transl, 0, 12, all_transl.slice(12 * fr, 12 * (fr + 1)))
				var cl := rd.compute_list_begin()
				chain.call(cl, 0, false, false)
				rd.compute_list_end()
			rd.submit()
			rd.sync()
			best = minf(best, float(Time.get_ticks_usec() - t0) / 1000.0 / frames.size())
		report += "GPU %s %d frames: %.3f ms/frame (one submission, wall clock incl. pose upload, best of 5) device=%s\n" % [clip, frames.size(), best, rd.get_device_name()]
	var tf := FileAccess.open(out + "/timing.txt", FileAccess.WRITE)
	tf.store_string(report)
	tf.close()
	print(report)
	print("DONE soma gpu")
	return 0
