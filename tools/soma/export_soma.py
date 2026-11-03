# One-time setup for the SOMA-X GPU pose kernels (kernels/anny, soma_*), and their oracle.
#
#   python tools/soma/export_soma.py <SOMA-X assets dir> <out dir> <clip dir>...
#
# Builds SOMA-X's SOMALayer with the ANNY identity and its pose correctives, on the CPU (setup and
# oracle only; the per-frame path is the kernels), and writes:
#
#   <out>/tables/*.bin + <out>/meta.json   every static table the kernels read, raw little-endian
#   <out>/clips/<clip>/                    per clip: the kernel inputs for every frame, and SOMA-X's
#                                          own vertices and world transforms for the
#                                          sampled frames (with and without correctives)
#
# The identity is the one tools/bridges/body_frames.py poses (mmg-bridges): every phenotype 0.5,
# gender 0. Kimodo's 30 joints map onto the public SOMA joints by name as body_frames.py maps them.
import json
import math
import os
import sys

import numpy as np
import torch

import soma.identity_model as _im
from soma import SOMALayer
from soma import procedural_transforms as _pt
from soma.geometry.rig_utils import joint_world_to_local
from soma.geometry.transforms import (
    matrix_to_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_xyzw_to_matrix,
)

NONE = 0xFFFFFFFF
MODES = ("local_x_euler", "local_x_swing_twist", "aligned_x_swing_twist")
STEP = 5        # sample every 5th frame: 60 frames over a 300-frame clip
NSAMPLE = 60

_fwd = _im.AnnySimplified.forward


def _fwd_dict(self, phenotype_kwargs=None, local_changes_kwargs=None):
    # SOMA-X passes its labels as a tensor; anny 0.2.0 parses tensors against its own labels.
    if isinstance(phenotype_kwargs, torch.Tensor):
        phenotype_kwargs = {k: phenotype_kwargs[:, i] for i, k in enumerate(self.phenotype_labels)}
    return _fwd(self, phenotype_kwargs, local_changes_kwargs if local_changes_kwargs is not None else {})


_im.AnnySimplified.forward = _fwd_dict


def put(d, name, arr, dtype, files):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(arr), dtype=dtype)
    a.tofile(os.path.join(d, name + ".bin"))
    files[name] = {"dtype": np.dtype(dtype).name, "shape": list(a.shape)}


def affine12(T):
    """(..., 4, 4) -> (..., 12): the 3x4 rows."""
    T = np.asarray(T)
    return T[..., :3, :4].reshape(*T.shape[:-2], 12)


def rounds_for(depth):
    return 0 if depth <= 0 else int(math.ceil(math.log2(depth + 1)))


def build_layer(assets):
    layer = SOMALayer(data_root=assets, identity_model_type="anny", device="cpu",
                      correctives_model_path=os.path.join(assets, "correctives_model.pt"))
    labels = list(layer.identity_model.identity_model.phenotype_labels)
    coeffs = torch.full((1, len(labels)), 0.5)
    if "gender" in labels:
        coeffs[0, labels.index("gender")] = 0.0
    layer.prepare_identity(coeffs)
    return layer


def pose(layer, poses, transl, correctives):
    with torch.no_grad():
        return layer.pose(torch.tensor(poses, dtype=torch.float32), transl=torch.tensor(transl, dtype=torch.float32),
                          pose2rot=False, apply_correctives=correctives)


def soma_internals(layer, poses, transl):
    """SOMA-X's own intermediates for these frames, through its own functions, after a pose() call
    has rebound the skinning state (rest shape and bind transforms)."""
    pt = layer.procedural_transforms
    bs = layer.batched_skinning
    with torch.no_grad():
        p = layer._pad_poses(torch.tensor(poses, dtype=torch.float32))
        rabs = pt.apply_source_joint_orient(p)
        pw = bs.forward_source_kinematics(local_rotations=rabs, global_translation=torch.tensor(transl, dtype=torch.float32),
                                          absolute_pose=True)
        tw = bs.expand_source_world_transforms(source_rotations=rabs, source_world_transforms=pw,
                                               transform_expander=pt.expand_world_transforms_from_source_fk)
    return rabs, pw, tw


def soma_channels(pt, rabs, pw):
    """SOMA-X's twist channels: Euler (3 channels) and swing-twist about each axis from the local
    rotations, and the aligned channel from the public world. Shapes (B,3,J), (B,3,J), (B,J)."""
    with torch.no_grad():
        J = rabs.shape[1]
        euler = _pt._local_euler_xyz_from_matrix(rabs).permute(0, 2, 1)
        ones = torch.ones(J)
        swing = torch.stack([_pt._swing_twist_channels_from_matrix(rabs, torch.full((J,), a, dtype=torch.long), ones)
                             for a in range(3)], dim=1)
        aligned = pt._aligned_twist_channels_from_world(pw)
    return euler.numpy(), swing.numpy(), aligned.numpy()


def two_atan2_channels(pt, rabs, pw):
    """The same channels by 2*atan2(q_a, q_w) (the full-angle swing-twist form), for the report:
    SOMA-X's half-angle form 4*atan2(q_a, 1 + q_w) differs from it once the rotation has swing."""
    with torch.no_grad():
        q = matrix_to_quaternion_xyzw(rabs)
        swing = torch.stack([2 * torch.atan2(q[..., a], q[..., 3]) for a in range(3)], dim=1)
        bq = pt.source_bind_quaternions
        aq = pt.segment_bind_align_quaternions
        vq = _pt._aligned_virtual_quaternions(pw[..., :3, :3], bq, aq, pt.aligned_virtual_segment_ids,
                                              pt.aligned_virtual_joint_ids)
        n = pt.segment_source_ids.numel()
        qe, qs, qp = vq[:, :n], vq[:, n:2 * n], vq[:, 2 * n:]

        def ang(a, b):
            r = quaternion_normalize_xyzw(quaternion_multiply_xyzw(quaternion_conjugate_xyzw(a), b))
            r = torch.where(r[..., 3:] < 0, -r, r)
            return 2 * torch.atan2(r[..., 0], r[..., 3])

        al = torch.zeros(pw.shape[0], pw.shape[1])
        al[:, pt.segment_end_source_ids] = ang(qs, qe)
        rm = pt.segment_reverse_mask
        if rm.any():
            al[:, pt.segment_start_source_ids[rm]] = ang(qp, qs)[:, rm]
    return swing.numpy(), al.numpy()


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def export_tables(layer, out):
    pt = layer.procedural_transforms
    bs = layer.batched_skinning
    cm = layer.correctives_model
    m = cm.module if hasattr(cm, "module") else cm
    d = os.path.join(out, "tables")
    os.makedirs(d, exist_ok=True)
    files = {}
    names = list(layer.public_joint_names)
    NJ = len(names)
    NT = bs.num_joints
    meta = {"NJ": NJ, "NT": NT}
    if layer.identity_lod_transfer is not None:
        raise SystemExit("identity_lod_transfer is set: the kernels do not carry it yet")
    if layer._pose_batch_bone_scales(1, torch.float32, "cpu") is not None:
        raise SystemExit("bone scales are set: the kernels do not carry them yet")

    # Public rig: parents (none for SOMA-X's self-parented root), bind-local translations, orient.
    src_par = list(bs.source_joint_parent_ids)
    pub_parent = [NONE if src_par[j] == j else src_par[j] for j in range(NJ)]
    put(d, "pub_parent", pub_parent, np.uint32, files)
    put(d, "pub_local_t", bs.source_local_translations[0] if bs.source_local_translations.ndim == 3
        else bs.source_local_translations, np.float32, files)
    put(d, "orient", pt.source_joint_orient, np.float32, files)
    put(d, "orient_pt", pt.source_joint_orient_parent_t, np.float32, files)
    put(d, "corr_bind", m.bindpose, np.float32, files)
    meta["gtj"] = int(bs.source_global_translation_joint_idx)

    def depth(par, j):
        n = 0
        while par[j] != NONE:
            j = par[j]
            n += 1
        return n
    meta["rounds_pub"] = rounds_for(max(depth(pub_parent, j) for j in range(NJ)))

    # Twist channels.
    put(d, "src_axis", pt.source_twist_axis_ids, np.uint32, files)
    put(d, "src_sign", pt.source_twist_axis_signs, np.float32, files)
    put(d, "seg_start", pt.segment_start_source_ids, np.uint32, files)
    put(d, "seg_end", pt.segment_end_source_ids, np.uint32, files)
    put(d, "seg_parent", pt.segment_parent_source_ids, np.uint32, files)
    put(d, "seg_rev", pt.segment_reverse_mask.to(torch.int64), np.uint32, files)
    # The rotations SOMA-X keeps as quaternions, as the 3x3 matrices they stand for (setup only).
    put(d, "bind_rot", quaternion_xyzw_to_matrix(pt.source_bind_quaternions), np.float32, files)
    put(d, "align_rot", quaternion_xyzw_to_matrix(pt.segment_bind_align_quaternions), np.float32, files)
    meta["NS"] = int(pt.segment_start_source_ids.numel())

    # Target rig: control copy, twist joints, fill.
    NTW = len(pt.twist_joint_names)
    meta["NTW"] = NTW
    twM = np.zeros((3, NTW, NJ), np.float32)
    for mi, mode in enumerate(pt.rotation_extraction_mode_names):
        twM[MODES.index(mode)] = pt.rotation_parameter_matrices_by_mode[mi].numpy()
    meta["modes"] = list(pt.rotation_extraction_mode_names)
    put(d, "twist_m", twM, np.float32, files)
    if pt._single_twist_axis is None:
        raise SystemExit("twist joints on several axes: SOMA-X uses Rodrigues there; check the cos/sin form first")
    put(d, "tw_axis", pt.twist_axis_ids, np.uint32, files)
    put(d, "tw_sign", pt.twist_axis_signs, np.float32, files)
    tpar = list(bs.joint_parent_ids)
    ctrl_t = pt.control_target_ids.tolist()
    ctrl_s = pt.control_source_ids.tolist()
    kind = np.full(NT, 2, np.uint32)
    ref = np.zeros(NT, np.uint32)
    par = np.array([NONE if tpar[t] == t else tpar[t] for t in range(NT)], np.uint32)
    t2s = {}
    for s_, t in zip(ctrl_s, ctrl_t):
        kind[t], ref[t], par[t] = 0, s_, NONE
        t2s[t] = s_
    for r, (t, tp) in enumerate(zip(pt.twist_target_ids.tolist(), pt.twist_parent_target_ids.tolist())):
        if tp not in t2s:
            raise SystemExit(f"twist joint {t}'s parent {tp} is not public: SOMA-X would read an identity there")
        kind[t], ref[t], par[t] = 1, r, t2s[tp]
    put(d, "tgt_kind", kind, np.uint32, files)
    put(d, "tgt_ref", ref, np.uint32, files)
    put(d, "tgt_par", par, np.uint32, files)
    put(d, "tgt_base_rot", bs.local_rotations[0], np.float32, files)
    put(d, "tgt_local_t", bs.local_translations[0], np.float32, files)

    def fill_len(t):
        n = 0
        while kind[t] == 2:
            n += 1
            t = tpar[t]
        return n
    meta["rounds_tgt"] = rounds_for(max(fill_len(t) for t in range(NT)))
    meta["n_fill"] = int((kind == 2).sum())

    # Skinning: inverse bind (what rebind keeps) and the top-8 renormalised weights SOMA-X's warp
    # skinning uses, as a dense V x NT matrix for anny_lbs.
    _, inv = joint_world_to_local(layer._cached_bind_transforms_world, bs.joint_parent_ids, return_inverse=True)
    put(d, "inv_bind", affine12(inv[0].detach().numpy()), np.float32, files)
    rest = layer._cached_rest_shape[0].detach().numpy()
    V = rest.shape[0]
    meta["V"] = V
    put(d, "rest", rest, np.float32, files)
    idx = bs.get_bone_indices().numpy()
    w = bs.get_bone_weights().numpy()
    W = np.zeros((V, NT), np.float32)
    for k in range(idx.shape[1]):
        ok = idx[:, k] >= 0
        W[np.nonzero(ok)[0], idx[ok, k]] += w[ok, k]
    put(d, "weights", W, np.float32, files)

    # Correctives: masked W1 (dense D x K) and masked W2 by vertex (CSR, 3 values per entry).
    W1 = (m.W1 * m.M1_prior).detach().numpy() if m.M1_prior is not None else m.W1.detach().numpy()
    W2 = (m.W2 * m.M2_prior).detach() if m.M2_prior is not None else m.W2.detach()
    gs = layer._cached_global_scale
    if isinstance(gs, torch.Tensor) or gs != 1.0:
        W2 = W2 * float(gs)   # SOMA-X scales the output by gs only when gs != 1
    meta["D"], meta["K"] = int(W1.shape[0]), int(W1.shape[1])
    meta["use_tanh"] = int(bool(m.use_tanh))
    put(d, "w1", W1, np.float32, files)
    W2v = W2.reshape(W2.shape[0], V, 3).permute(1, 0, 2)              # (V, K, 3)
    nz = (W2v != 0).any(dim=2)                                         # (V, K)
    rows, cols = torch.nonzero(nz, as_tuple=True)
    rowptr = np.zeros(V + 1, np.uint32)
    np.add.at(rowptr, rows.numpy() + 1, 1)
    rowptr = np.cumsum(rowptr).astype(np.uint32)
    put(d, "w2_rowptr", rowptr, np.uint32, files)
    put(d, "w2_col", cols.numpy(), np.uint32, files)
    put(d, "w2_val", W2v[rows, cols].numpy(), np.float32, files)
    meta["E"] = int(cols.numel())
    meta["files"] = files
    return meta


def clip_poses(layer, d):
    names = list(layer.public_joint_names)
    par = layer.output_joint_parent_ids.tolist()
    par[0] = -1
    J = len(names)
    kn = [n.strip() for n in open(f"{d}/skeleton.txt").read().split("\n")[3:] if n.strip()]
    kp = [int(x) for x in open(f"{d}/parents.txt").read().split()]
    data = np.fromfile(f"{d}/transforms.f32", np.float32).reshape(-1, len(kn), 12).astype(np.float64)
    F = len(data)
    poses = np.tile(np.eye(3), (F, J - 1, 1, 1))
    for kj, n in enumerate(kn):
        if n not in names:
            continue
        sj = names.index(n)
        if kp[kj] >= 0 and par[sj] != names.index(kn[kp[kj]]):
            continue
        poses[:, sj - 1] = data[:, kj, :9].reshape(F, 3, 3)
    rest = pose(layer, np.tile(np.eye(3), (1, J - 1, 1, 1)), np.zeros((1, 3)), False)
    hips_h = float(rest["transforms"][0, 1, 1, 3].item() - rest["vertices"][0, :, 1].min().item())
    transl = data[:, 0, 9:12] * (hips_h / data[0, 0, 10])
    return poses.astype(np.float32), transl.astype(np.float32)


def export_clip(layer, d, out, report):
    key = os.path.basename(d.rstrip("/"))
    cd = os.path.join(out, "clips", key)
    os.makedirs(cd, exist_ok=True)
    files = {}
    poses, transl = clip_poses(layer, d)
    F = len(poses)
    frames = np.arange(0, STEP * NSAMPLE, STEP)
    assert frames[-1] + 1 < F
    J = poses.shape[1] + 1
    padded = np.concatenate([np.tile(np.eye(3, dtype=np.float32), (F, 1, 1, 1)), poses], axis=1)
    put(cd, "pose", padded.reshape(F, J * 9), np.float32, files)
    put(cd, "transl", transl, np.float32, files)
    put(cd, "frames", frames, np.uint32, files)
    on = pose(layer, poses[frames], transl[frames], True)["vertices"].numpy()
    off = pose(layer, poses[frames], transl[frames], False)["vertices"].numpy()
    put(cd, "verts_on", on, np.float32, files)
    put(cd, "verts_off", off, np.float32, files)
    rabs, pw, tw = soma_internals(layer, poses[frames], transl[frames])
    put(cd, "world", affine12(tw.numpy()), np.float32, files)
    put(cd, "pub_world", affine12(pw.numpy()), np.float32, files)
    _, swing, aligned = soma_channels(layer.procedural_transforms, rabs, pw)
    s2, a2 = two_atan2_channels(layer.procedural_transforms, rabs, pw)
    report[key] = {
        "corrective_mm_max": float(np.linalg.norm(on - off, axis=-1).max() * 1000),
        "two_atan2_vs_soma_swing_rad_max": float(np.abs(wrap(s2 - swing)).max()),
        "two_atan2_vs_soma_aligned_rad_max": float(np.abs(wrap(a2 - aligned)).max()),
    }
    json.dump({"F": F, "files": files}, open(os.path.join(cd, "meta.json"), "w"), indent=1)
    print(f"CLIP {key} F={F} sampled={len(frames)} {report[key]}", flush=True)


def main(argv):
    assets, out, clips = argv[1], argv[2], argv[3:]
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    layer = build_layer(assets)
    # One pose() call first, so the skinning state is the one every pose() rebinds to.
    pose(layer, np.tile(np.eye(3), (1, len(layer.public_joint_names) - 1, 1, 1)), np.zeros((1, 3)), True)
    meta = export_tables(layer, out)
    report = {}
    for d in clips:
        export_clip(layer, d, out, report)
    meta["report"] = report
    meta["clips"] = [os.path.basename(d.rstrip("/")) for d in clips]
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("META", {k: v for k, v in meta.items() if k not in ("files", "report")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
