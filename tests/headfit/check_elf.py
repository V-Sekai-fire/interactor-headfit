"""Checks headfit.elf before a viewer gets it: a RISC-V 64 ELF whose .sandbox_variant section names
the 40-byte Variant of a double-precision Godot, and whose API names the viewer's calls.

    python3 tests/headfit/check_elf.py headfit.elf

Exit 0 when every check holds; each failure is printed.
"""
import struct
import sys

API = ("hf_mesh_create", "hf_mesh_set_roles", "hf_anime_shape", "hf_error", "hf_destroy")


def sections(b):
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    heads = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + k * shentsize) for k in range(shnum)]
    names = heads[shstrndx]
    out = {}
    for h in heads:
        end = b.index(b"\0", names[4] + h[0])
        out[b[names[4] + h[0]:end].decode()] = b[h[4]:h[4] + h[5]]
    return out


def main(path):
    b = open(path, "rb").read()
    bad = []
    if b[:4] != b"\x7fELF" or b[4] != 2 or struct.unpack_from("<H", b, 0x12)[0] != 243:
        bad.append("not a 64-bit RISC-V ELF")
    else:
        s = sections(b)
        v = s.get(".sandbox_variant", b"")
        if len(v) < 8 or v[:4] != b"SBXV":
            bad.append("no .sandbox_variant section: the viewer takes it for a 24-byte Variant and refuses it")
        elif struct.unpack_from("<I", v, 4)[0] != 40:
            bad.append("the Variant is %d bytes; the double-precision viewer's is 40" % struct.unpack_from("<I", v, 4)[0])
        for name in API:
            if name.encode() + b"\0" not in b:
                bad.append("the API has no " + name)
    for line in bad:
        print("FAIL", line)
    print("headfit.elf %s: %d bytes, %s" % (path, len(b), "PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
