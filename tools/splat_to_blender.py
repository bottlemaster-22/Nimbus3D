"""Turn a 3D Gaussian Splatting .ply into one Blender can actually colour.

WHY THIS EXISTS. Our exported .ply, and every other 3DGS .ply, stores colour as
`f_dc_0/1/2`: raw degree-0 spherical-harmonic coefficients, which are signed and
roughly in the range -1.8..1.8. Blender's .ply importer only understands vertex
colour under the names `red green blue` as bytes. It sees no property it
recognises, imports the points with no colour at all, and draws them as black
dots. Nothing is wrong with the file.

The conversion is the INRIA display formula this repo already uses in
SplatCloud.shDCToColor:

    colour = 0.5 + 0.282095 * f_dc

plus a sigmoid on `opacity`, which is stored as a logit, so nearly invisible
splats can be dropped rather than muddying the cloud.

It also rotates into Blender's frame. Our PLY writer emits RDF (X right, Y down,
Z front - the INRIA reference convention, see PLYCodec), while Blender is Z-up:
(x, y, z) -> (x, z, -y).

What is DISCARDED, deliberately: scale, rotation and the higher-degree SH bands.
This produces a coloured POINT CLOUD for looking at and for lining up against
other geometry. It is not a splat renderer and will not look like the viewer.
For actual splat rendering in Blender, use the 3DGS Render add-on.

    python tools/splat_to_blender.py scan.ply
    python tools/splat_to_blender.py scan.ply out.ply --min-alpha 0.1 --keep-frame
"""

import argparse
import struct
import sys

SH_DC_TO_COLOR = 0.282095017


def read_header(f):
    """Returns (count, properties, little_endian). Leaves f at the data."""
    magic = f.readline().strip()
    if magic != b"ply":
        raise SystemExit("not a .ply file (no 'ply' magic on line 1)")
    count = None
    props = []          # list of (type, name), in file order
    little = True
    in_vertex = False
    while True:
        line = f.readline()
        if not line:
            raise SystemExit("header never ended: no 'end_header' line")
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == b"format":
            fmt = parts[1]
            if fmt == b"ascii":
                raise SystemExit("ascii .ply is not supported; export binary")
            little = fmt == b"binary_little_endian"
        elif key == b"element":
            in_vertex = parts[1] == b"vertex"
            if in_vertex:
                count = int(parts[2])
        elif key == b"property" and in_vertex:
            if parts[1] == b"list":
                raise SystemExit("list properties on vertices are not supported")
            props.append((parts[1].decode(), parts[2].decode()))
        elif key == b"end_header":
            break
    if count is None:
        raise SystemExit("no 'vertex' element in the header")
    return count, props, little


# ply type name -> struct code and byte width
TYPES = {
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--min-alpha", type=float, default=0.05,
                    help="drop splats fainter than this after the sigmoid "
                         "(0 keeps everything, default 0.05)")
    ap.add_argument("--keep-frame", action="store_true",
                    help="skip the rotation into Blender's Z-up frame")
    args = ap.parse_args()

    dest = args.dest or args.source.rsplit(".", 1)[0] + "_blender.ply"

    with open(args.source, "rb") as f:
        count, props, little = read_header(f)
        names = [n for _, n in props]
        for needed in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
            if needed not in names:
                raise SystemExit(
                    "%s has no '%s' property, so it is not a 3DGS .ply. "
                    "Properties found: %s" % (args.source, needed,
                                              ", ".join(names[:12])))

        codes = ""
        for t, _ in props:
            if t not in TYPES:
                raise SystemExit("unknown property type '%s'" % t)
            codes += TYPES[t][0]
        row = struct.Struct(("<" if little else ">") + codes)
        idx = {n: i for i, n in enumerate(names)}
        ix, iy, iz = idx["x"], idx["y"], idx["z"]
        ic = (idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"])
        io_ = idx.get("opacity")

        blob = f.read(row.size * count)
        if len(blob) < row.size * count:
            raise SystemExit("file ends early: wanted %d vertices, got %d"
                             % (count, len(blob) // row.size))

    out = bytearray()
    pack = struct.Struct("<fffBBBB").pack
    kept = 0
    for v in row.iter_unpack(blob):
        if io_ is None:
            a = 1.0
        else:
            o = v[io_]
            # sigmoid, written so a large negative logit cannot overflow exp
            a = 1.0 / (1.0 + 2.718281828459045 ** -o) if o > -60.0 else 0.0
        if a < args.min_alpha:
            continue
        x, y, z = v[ix], v[iy], v[iz]
        if not args.keep_frame:
            x, y, z = x, z, -y
        rgb = []
        for i in ic:
            c = 0.5 + SH_DC_TO_COLOR * v[i]
            rgb.append(0 if c <= 0.0 else (255 if c >= 1.0 else int(c * 255.0 + 0.5)))
        out += pack(x, y, z, rgb[0], rgb[1], rgb[2],
                    int(a * 255.0 + 0.5))
        kept += 1

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment converted by tools/splat_to_blender.py for Blender import\n"
        "element vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar alpha\n"
        "end_header\n" % kept
    )
    with open(dest, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out)

    dropped = count - kept
    print("%s\n  %d of %d points kept (%d below alpha %.2f)"
          % (dest, kept, count, dropped, args.min_alpha))


if __name__ == "__main__":
    main()
