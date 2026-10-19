# headfit.elf, the godot-sandbox guest, from this checkout and the goal manifest's public sibling
# checkouts under WEFT_ROOT:
#   2-contract/guest-runtime  the guest API (vendor/sandbox-api) and slang-rt's prelude
#   2-contract/guest-common   guest/common/hf_geom
#   2-contract/lbfgsb         guest/drape/lbfgsb, vec_cpu
#   2-contract/sinew-align    vendor/sinew-align
#   3-interactor/drape        guest/drape/similarity.h, primitives.h
#
#   cmake -S guest -B build/rv64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=<riscv64-sysroot>/toolchain.cmake \
#       -DWEFT_ROOT=<manifest root>
#   cmake --build build/rv64 --target headfit
#
# The viewer is a double-precision Godot, so the guest API is built with DOUBLE_PRECISION: its
# .sandbox_variant section names the 40-byte Variant, and a single-precision build is refused at load.
# The fit's sources are compiled uncontracted (-ffp-contract=off), as tests/headfit's native runner is.
cmake_minimum_required(VERSION 3.16)
project(headfit_guest C CXX)
if(CMAKE_C_COMPILER_TARGET AND NOT CMAKE_ASM_COMPILER_TARGET)
	set(CMAKE_ASM_COMPILER_TARGET ${CMAKE_C_COMPILER_TARGET})
endif()
enable_language(ASM)

set(WEFT_ROOT "" CACHE PATH "The goal manifest's root (holds 2-contract/ and 3-interactor/)")
if(NOT WEFT_ROOT)
	message(FATAL_ERROR "set -DWEFT_ROOT=<manifest root>")
endif()
get_filename_component(HF_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GR "${WEFT_ROOT}/2-contract/guest-runtime")
set(GC "${WEFT_ROOT}/2-contract/guest-common")
set(LB "${WEFT_ROOT}/2-contract/lbfgsb")
set(SA "${WEFT_ROOT}/2-contract/sinew-align")
set(DR "${WEFT_ROOT}/3-interactor/drape")
foreach(f "${GR}/vendor/sandbox-api/cmake/CMakeLists.txt" "${GC}/guest/common/hf_geom.cpp" "${LB}/guest/drape/lbfgsb.cpp"
		"${SA}/vendor/sinew-align/sinew_align.c" "${DR}/guest/drape/similarity.h")
	if(NOT EXISTS "${f}")
		message(FATAL_ERROR "missing ${f}: clone the sibling checkouts under WEFT_ROOT")
	endif()
endforeach()

# Sources are named relative to the checkouts, so the ELF names no machine or folder.
add_compile_options("-ffile-prefix-map=${WEFT_ROOT}/=" "-ffile-prefix-map=${HF_ROOT}/=")
set(DOWNLOAD_RUNTIME_API OFF CACHE BOOL "" FORCE)
set(DOUBLE_PRECISION ON CACHE BOOL "" FORCE)
add_subdirectory("${GR}/vendor/sandbox-api/cmake" sandbox_api)

set(HF_FIT_SOURCES "${HF_ROOT}/guest/headfit/hf_core.cpp" "${HF_ROOT}/guest/headfit/hf_kernels.cpp"
	"${GC}/guest/common/hf_geom.cpp" "${SA}/vendor/sinew-align/sinew_align.c")
add_sandbox_program(headfit "${HF_ROOT}/guest/headfit/main.cpp" ${HF_FIT_SOURCES}
	"${LB}/guest/drape/lbfgsb.cpp" "${LB}/guest/drape/vec_cpu.cpp")
set_source_files_properties(${HF_FIT_SOURCES} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(headfit PRIVATE "${HF_ROOT}/guest/headfit" "${GC}/guest" "${LB}/guest/drape" "${DR}/guest/drape"
	"${GR}/guest/avbd/slang-rt" "${SA}/vendor/sinew-align")
target_link_libraries(headfit PRIVATE -lm -lstdc++)
