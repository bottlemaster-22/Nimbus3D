# =============================================================================
#  ply_reader.py -- generic PLY reader for 3D Gaussian Splat point clouds.
#
#  This does NOT assume a fixed property list. It parses whatever `element
#  vertex` properties the header declares, in the order declared, and
#  returns them as a dict of {property_name: array.array}. splat_data.py is
#  the layer that knows which property NAMES mean "this is a Gaussian
#  splat" and maps them onto position / scale / rotation / color / opacity.
#
#  Why not assume a fixed layout: this add-on was first written before
#  docs/DATA_FORMAT.md and ios/Sources/Export/PLYCodec.swift existed on
#  disk, so it was built to not need them -- and now that both exist and
#  have been checked (see README.md "Data formats"), the property names and
#  quantisation already match, but parsing the header generically remains
#  the right design regardless: it means this importer works against the
#  de-facto INRIA/gsplat PLY convention used by every public 3DGS tool AND
#  against our own exporter, as long as it emits standard PLY -- no
#  re-write needed either way.
#
#  Supports: ascii, binary_little_endian, binary_big_endian (PLY format
#  spec, all three are legal). Only `element vertex` is read; any other
#  element in the file (e.g. `element face`) is skipped correctly by
#  computing its byte stride from its own property list, so extra elements
#  never desync the parse.
# =============================================================================

import array
import struct

# PLY scalar type name -> (struct format char, byte size)
_TYPE_MAP = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}

# array.array typecode per struct format char (for fast bulk decode of the
# common case: a list-free scalar property, which is every property a 3DGS
# ply uses).
_ARRAY_TYPECODE = {
    "b": "b", "B": "B", "h": "h", "H": "H",
    "i": "i", "I": "I", "f": "f", "d": "d",
}


class PlyParseError(Exception):
    """Raised for a file that is not a well-formed PLY, or whose vertex
    element uses a feature this reader intentionally does not support
    (list properties on the vertex element -- not used by any 3DGS
    exporter, ascii element other than vertex, etc.)."""


class PlyElement:
    __slots__ = ("name", "count", "properties")

    def __init__(self, name, count):
        self.name = name
        self.count = count
        # list of (prop_name, type_char, size, is_list, list_count_type)
        self.properties = []


def _read_header(f):
    magic = f.readline().strip()
    if magic != b"ply":
        raise PlyParseError(f"not a PLY file (magic line was {magic!r})")

    fmt = None
    elements = []
    current = None

    while True:
        line = f.readline()
        if not line:
            raise PlyParseError("PLY header never terminated with end_header")
        line = line.strip()
        if not line or line.startswith(b"comment") or line.startswith(b"obj_info"):
            continue
        tokens = line.split()
        keyword = tokens[0]

        if keyword == b"format":
            fmt = tokens[1].decode("ascii")
        elif keyword == b"element":
            name = tokens[1].decode("ascii")
            count = int(tokens[2])
            current = PlyElement(name, count)
            elements.append(current)
        elif keyword == b"property":
            if current is None:
                raise PlyParseError("property line before any element line")
            if tokens[1] == b"list":
                count_type = tokens[2].decode("ascii")
                val_type = tokens[3].decode("ascii")
                prop_name = tokens[4].decode("ascii")
                if val_type not in _TYPE_MAP or count_type not in _TYPE_MAP:
                    raise PlyParseError(f"unsupported list property type in {line!r}")
                current.properties.append((prop_name, val_type, None, True, count_type))
            else:
                type_name = tokens[1].decode("ascii")
                prop_name = tokens[2].decode("ascii")
                if type_name not in _TYPE_MAP:
                    raise PlyParseError(f"unsupported property type {type_name!r}")
                current.properties.append((prop_name, type_name, None, False, None))
        elif keyword == b"end_header":
            break
        # anything else (unknown keyword) is ignored per PLY tolerance norms

    if fmt is None:
        raise PlyParseError("PLY header had no 'format' line")
    return fmt, elements


def _vertex_element(elements):
    for el in elements:
        if el.name == "vertex":
            return el
    raise PlyParseError("PLY file has no 'element vertex' -- not a point cloud")


def read_ply_vertices(filepath):
    """Parse a PLY file and return (count, {property_name: array.array}) for
    the `vertex` element only. Raises PlyParseError on anything this reader
    does not support.
    """
    with open(filepath, "rb") as f:
        fmt, elements = _read_header(f)
        vertex_el = _vertex_element(elements)

        if any(p[3] for p in vertex_el.properties):
            raise PlyParseError(
                "vertex element has a list property; no known 3DGS PLY "
                "exporter does this -- refusing to guess"
            )

        if fmt == "ascii":
            return _read_ascii_body(f, elements, vertex_el)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt == "binary_little_endian" else ">"
            return _read_binary_body(f, elements, vertex_el, endian)
        else:
            raise PlyParseError(f"unrecognised PLY format {fmt!r}")


def _read_binary_body(f, elements, vertex_el, endian):
    result = {name: array.array(_ARRAY_TYPECODE[_TYPE_MAP[tname][0]])
              for (name, tname, _, is_list, _c) in vertex_el.properties}

    for el in elements:
        if el is vertex_el:
            # Fast path: fixed-stride scalar properties only (true of every
            # known 3DGS PLY exporter). Read the whole element in one go
            # per property via struct, column-major, by reading row-major
            # bytes and unpacking per row -- simplest correct approach that
            # still avoids a Python-level loop per SCALAR (we loop per row,
            # unavoidable without numpy, but each row is one struct.unpack).
            row_format = endian + "".join(
                _TYPE_MAP[tname][0] for (_n, tname, _s, _l, _c) in vertex_el.properties
            )
            row_size = struct.calcsize(row_format)
            prop_names = [p[0] for p in vertex_el.properties]
            unpack = struct.Struct(row_format).unpack
            raw = f.read(row_size * el.count)
            if len(raw) != row_size * el.count:
                raise PlyParseError(
                    f"PLY truncated: expected {row_size * el.count} bytes "
                    f"for {el.count} vertices, got {len(raw)}"
                )
            for i in range(el.count):
                row = unpack(raw[i * row_size:(i + 1) * row_size])
                for name, value in zip(prop_names, row):
                    result[name].append(value)
        else:
            # Skip any other element (e.g. 'face') correctly by computing
            # its row size and seeking past it. List properties on a
            # non-vertex element (faces) are legal PLY; walk them row by
            # row rather than assuming a fixed stride.
            _skip_element(f, el, endian)

    return vertex_el.count, result


def _skip_element(f, el, endian):
    has_list = any(p[3] for p in el.properties)
    if not has_list:
        row_format = endian + "".join(_TYPE_MAP[t][0] for (_n, t, _s, _l, _c) in el.properties)
        f.seek(struct.calcsize(row_format) * el.count, 1)
        return
    # General case (e.g. face list): walk row by row.
    for _ in range(el.count):
        for (_name, tname, _size, is_list, count_type) in el.properties:
            if is_list:
                (count_char, count_size) = _TYPE_MAP[count_type]
                n = struct.unpack(endian + count_char, f.read(count_size))[0]
                (val_char, val_size) = _TYPE_MAP[tname]
                f.seek(val_size * n, 1)
            else:
                (val_char, val_size) = _TYPE_MAP[tname]
                f.seek(val_size, 1)


def _read_ascii_body(f, elements, vertex_el):
    result = {name: array.array(_ARRAY_TYPECODE[_TYPE_MAP[tname][0]])
              for (name, tname, _, is_list, _c) in vertex_el.properties}

    for el in elements:
        if el is vertex_el:
            prop_defs = vertex_el.properties
            for _ in range(el.count):
                line = f.readline()
                if not line:
                    raise PlyParseError("ascii PLY truncated before all vertices read")
                tokens = line.split()
                for (name, tname, _s, _l, _c), tok in zip(prop_defs, tokens):
                    char = _TYPE_MAP[tname][0]
                    value = float(tok) if char in ("f", "d") else int(tok)
                    result[name].append(value)
        else:
            for _ in range(el.count):
                f.readline()  # ascii: one non-vertex element line per row (faces etc.)

    return vertex_el.count, result
