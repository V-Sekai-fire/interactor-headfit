# interactor-headfit

The head fit as a sandboxed guest program: it fits the parametric body model's head and transfers a shared facial-expression set onto it.

## What it is for

It fits the head of the parametric body model to a target and carries facial expressions across to it. Its kernels are emitted from Lean through a shader compiler, so the GPU and CPU paths are generated from one source. It is placed in the goal manifest and builds against its sibling checkouts there; `transport-meshing-pen` builds the guest binaries.

## Build and run

```sh
kernels/headfit/gen.sh
tests/headfit/build.sh
```

`gen.sh` regenerates the kernels, and `build.sh` builds the native check runner. `build.sh` reads `guest/drape`, `guest/avbd` and `vendor/sinew-align`, which this repository does not hold, so it currently builds only inside `transport-meshing-pen`.

The guest, `headfit.elf`, builds from this checkout and public sibling checkouts of the goal manifest (`guest/CMakeLists.txt` lists them), with the riscv64 sysroot's toolchain:

```sh
cmake -S guest -B build/rv64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=<riscv64-sysroot>/toolchain.cmake -DWEFT_ROOT=<manifest root>
cmake --build build/rv64 --target headfit
python3 tests/headfit/check_elf.py build/rv64/headfit
```

It is built for a double-precision Godot. The `guest` workflow builds it on every push and uploads it as the artifact `headfit-elf-<commit>`.

## Licence

MIT. See [LICENSE](LICENSE).
