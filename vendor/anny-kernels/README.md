# contract-anny-kernels

The ANNY body model's forward and backward kernels in Lean, emitted through Slang for the in-guest fits.

## What it is for

The Lean kernels restate ANNY's data-parallel stages, from blendshapes and the joint regressor to forward kinematics, skinning and the vertex residual, each with its vector-Jacobian product, so the in-guest L-BFGS-B can fit a body to a target mesh. They are emitted as Slang and compiled to C++ for the CPU path and SPIR-V for the GPU path. It finds the repositories it builds against as sibling checkouts at their paths in the goal manifest.

## Build

```sh
kernels/anny/gen.sh
```

## SOMA-X posing on the GPU

The `soma_*` kernels pose a SOMA-X body (ANNY identity, pose correctives) every frame on the GPU,
from the frame's joint rotations and hips translation to the skinned vertices:

| Kernel | Per frame |
|---|---|
| `soma_orient` | joint orient, the FK seeds of the public rig, the correctives' input features |
| `soma_jump` | one round of pointer-jumping FK (public rig, then the fill of the target rig) |
| `soma_twist` | the twist channels of SOMA-X's three extraction modes, from 3×3 matrices only |
| `soma_expand` | the target rig: copied public joints, twist joints, seeds for the rest |
| `soma_skin` | skinning matrices, world times inverse bind |
| `soma_corr_l1` | correctives hidden layer (masked W1, relu, tanh if the checkpoint uses it) |
| `soma_corr_l2` | correctives output layer (masked W2 as CSR by vertex) added to the rest shape |
| `anny_lbs` | linear blend skinning |

No kernel forms a quaternion. SOMA-X's swing-twist angle, `4·atan2(q_a, 1 + w)` on the half-angle
quaternion, is read off the matrix as `4·atan2(num_a, (1 + tr) + 2√(1 + tr))`. It is not the plain
swing-twist angle `2·atan2(q_a, w)`: on the test clips the two differ by up to 0.11 rad. The plain
split, the rank-1 Kabsch completion of the rotational-star solver
(contract-rotation-star `RotationStar.Slang.Split.swingTwistAngle`), is the default
(`twistForm = 0`); SOMA-X's form is kept behind `twistForm = 1` so the gap can be measured on
the mesh.

Setup runs once on the CPU: `tools/soma/export_soma.py` builds SOMA-X's layer and writes every
table the kernels read, plus SOMA-X's own vertices for the test clips. Then:

```sh
tests/soma_pose/build.sh <fixture>             # C++ emission against SOMA-X
GODOT=<godot 4> tests/soma_pose/gpu.sh <fixture>  # SPIR-V on a RenderingDevice, in a window
```

Both gate the vertices on a float32 rounding bound (stated in `tests/soma_pose/test.cpp`), and
check that three controls fail: correctives off, twist joints skipped, one frame late.
`tools/soma/reemit.sh` re-emits and re-pins the kernels after a change to their Lean.

Not yet carried: SOMA-X's `identity_lod_transfer`, per-identity bone scales, and twist joints on
several axes. The exporter stops if the identity needs any of them.

## Licence

MIT. See [LICENSE](LICENSE). The `CITATION.cff` files name the licences of the body model and the Lean tree the kernels derive from.
