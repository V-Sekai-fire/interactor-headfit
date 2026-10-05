# interactor-headfit

The head fit as a sandboxed guest program: it fits the parametric body model's head and transfers a shared facial-expression set onto it.

## What it is for

It fits the head of the parametric body model to a target and carries facial expressions across to it. Its kernels are emitted from Lean through a shader compiler, so the GPU and CPU paths run one source. It is placed in the goal manifest and builds against its sibling checkouts there; `transport-meshing-pen` builds the guest binaries.

## Build and run

```sh
kernels/headfit/gen.sh
tests/headfit/build.sh
```

`gen.sh` regenerates the kernels, and `build.sh` builds the native check runner.

## Licence

No LICENSE file is present. The sources carry `Apache-2.0 OR MIT` SPDX headers.
