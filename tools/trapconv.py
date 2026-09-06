"""Find TRAPPING float-to-integer conversions.

Swift's `Int64(someFloat)`, `Int32(...)`, `UInt8(...)` and friends are TRAPPING
conversions. They kill the process, in release as well as debug, on:

  * NaN
  * infinity
  * any finite value outside the destination type's range

This is not theoretical here. It is the crash the owner reported as
"crashing quite a bit, even with RAM free", and it was found in THREE separate
files, all written the same way:

    let ix = Int64((position.x / size).rounded(.down))

A single ARKit frame with an unusable camera transform produced a NaN pose,
every world position derived from it was NaN, and the first of roughly half a
million conversions per second killed the app. Memory was irrelevant, which is
exactly why "RAM was free" was such a good clue.

A SUBTLETY THIS TOOL EXISTS TO CATCH
Clamping does not necessarily save you, because Swift's min/max are:

    min(x, y) { y <  x ? y : x }
    max(x, y) { y >= x ? y : x }

NaN compares false against everything, so the SECOND argument is absorbed and
the FIRST propagates:

    UInt8(max(0, min(255, x)))          -> 255 when x is NaN.  SAFE.
    UInt8(min(max(x * 255, 0), 255))    -> NaN when x is NaN.  TRAPS.

Those two lines look the same and behave completely differently. That is not
something to rely on people noticing.

USAGE
    python tools/trapconv.py             report and exit non-zero on anything
                                         not in the allowlist
    python tools/trapconv.py --list      print every candidate, exit 0
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.join(REPO, 'ios', 'Sources')
ALLOWLIST = os.path.join(HERE, 'trapconv_allowlist.txt')

INT_TYPES = r'(?:Int|Int8|Int16|Int32|Int64|UInt|UInt8|UInt16|UInt32|UInt64)'

# A conversion call: Type( ... ). We capture the argument text by matching
# balanced parentheses manually, because a regex cannot.
CALL = re.compile(r'(?<![A-Za-z0-9_.])' + INT_TYPES + r'\(')

# Spellings that are explicitly NON-trapping and therefore always fine.
NON_TRAPPING = ('truncatingIfNeeded:', 'exactly:', 'clamping:', 'bitPattern:',
                'ascii:', 'radix:')

# Evidence in the argument that a float is involved.
FLOATY = re.compile(
    r'\.rounded\(|Float\(|Double\(|CGFloat\(|\bfloor\(|\bceil\(|/\s*[A-Za-z_]'
    r'|\.x\b|\.y\b|\.z\b|\.w\b|Meters\b|Metres\b|Seconds\b|fraction|Fraction'
    r'|sqrt|expf|logf|powf|sigmoid|\* *255|\* *100'
)

# Things that make an INTEGER argument obvious, so we do not flag Int(count).
INTY = re.compile(r'\.count\b|\.utf8\b|\bindex\b|\bcount\b|<<|>>|0x[0-9A-Fa-f]')


def strip_comments_and_strings(text):
    out, i, n = [], 0, len(text)
    line_c, block_c, in_str = False, 0, False
    while i < n:
        c, nxt = text[i], (text[i + 1] if i + 1 < n else '')
        if line_c:
            out.append('\n' if c == '\n' else ' ')
            if c == '\n':
                line_c = False
            i += 1
        elif block_c:
            if c == '*' and nxt == '/':
                block_c -= 1
                out.append('  '); i += 2
            elif c == '/' and nxt == '*':
                block_c += 1
                out.append('  '); i += 2
            else:
                out.append('\n' if c == '\n' else ' '); i += 1
        elif in_str:
            if c == '\\':
                # A backslash at the END OF A LINE is Swift's multi-line string
                # continuation, and blanking both characters deleted that
                # newline. Every candidate below it in the file was then
                # reported one line too high, and the allowlist entry written
                # from that report pointed at innocent code. Fifteen of these
                # in MetalSplatTrainer.swift put the tail of its report fifteen
                # lines out, which is how a line-numbered allowlist quietly
                # stops meaning anything.
                out.append(' ')
                out.append('\n' if nxt == '\n' else ' ')
                i += 2
            elif c == '"':
                in_str = False; out.append(' '); i += 1
            else:
                out.append('\n' if c == '\n' else ' '); i += 1
        else:
            if c == '/' and nxt == '/':
                line_c = True; out.append('  '); i += 2
            elif c == '/' and nxt == '*':
                block_c = 1; out.append('  '); i += 2
            elif c == '"':
                in_str = True; out.append(' '); i += 1
            else:
                out.append(c); i += 1
    return ''.join(out)


def argument_of(text, open_index):
    """Text between the parens starting at open_index, balanced."""
    depth, i, n = 0, open_index, len(text)
    while i < n:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
        i += 1
    return ''


def nan_absorbed(arg):
    """True when the OUTERMOST clamp puts the variable in a position Swift's
    min/max absorbs (second argument), so a NaN cannot reach the conversion."""
    a = arg.replace(' ', '')
    # max(0,min(255,x)) and min(255,max(0,x)) both absorb: the float is last.
    return bool(re.match(r'^(?:Swift\.)?(?:max|min)\(-?[0-9.]+,', a))


# Separator between the file and the exact source text that was triaged.
ENTRY_SEP = ' :: '


def entry_key(rel, text):
    """What an allowlist entry is keyed on.

    The file plus the SOURCE TEXT of the conversion, not the file plus a
    line number. Line numbers were the original key and they were wrong:
    inserting anything above a triaged line detached its entry and turned
    CI red on a file nobody had touched. That happened four separate times
    in one afternoon, twice by hand and twice caught by reviewers reading
    patches that would have done it again.

    The text is also the honest key. What a person read and judged was the
    conversion, not its position in the file, so moving a line should keep
    its verdict and CHANGING one should lose it. That is exactly what this
    key does, and the line-number key had it backwards on both counts.

    Two identical conversions in one file share an entry. That is correct:
    identical code in the same file has the same argument for or against
    it, and writing the reason twice would only create a chance to write it
    differently the second time.
    """
    return (rel, ' '.join(text.split()))


def load_allowlist():
    seen = set()
    if not os.path.exists(ALLOWLIST):
        return seen
    for raw in io.open(ALLOWLIST, encoding='utf-8'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key = line.split('  # ', 1)[0].strip()
        if ENTRY_SEP not in key:
            # A stale File:line entry. Not silently tolerated: it would
            # read as coverage that no longer exists.
            print('  allowlist entry is not in File%scode form: %s'
                  % (ENTRY_SEP, key))
            continue
        rel, text = key.split(ENTRY_SEP, 1)
        seen.add(entry_key(rel.strip(), text))
    return seen


def scan(root):
    hits = []
    for dirpath, _, names in os.walk(root):
        for fn in names:
            if not fn.endswith('.swift'):
                continue
            path = os.path.join(dirpath, fn)
            raw = io.open(path, encoding='utf-8', errors='replace').read()
            code = strip_comments_and_strings(raw)
            rawlines = raw.split('\n')
            for m in CALL.finditer(code):
                open_index = m.end() - 1
                arg = argument_of(code, open_index)
                if not arg.strip():
                    continue
                if any(k in arg for k in NON_TRAPPING):
                    continue
                if not FLOATY.search(arg):
                    continue
                if INTY.search(arg) and not re.search(r'\.rounded\(|Float\(', arg):
                    continue
                if nan_absorbed(arg):
                    continue
                line_no = code[:open_index].count('\n') + 1
                rel = os.path.relpath(path, root).replace('\\', '/')
                hits.append((rel, line_no, rawlines[line_no - 1].strip()[:110]))
    return hits


def main(argv):
    listing = '--list' in argv
    root = DEFAULT_ROOT
    hits = scan(root)
    allow = load_allowlist()

    untriaged = [h for h in hits if entry_key(h[0], h[2]) not in allow]

    print('trapconv: scanning %s' % root)
    print('  trapping float-to-int conversions found: %d' % len(hits))
    print('  triaged and allowlisted:                 %d' % (len(hits) - len(untriaged)))
    print('  NOT triaged:                             %d' % len(untriaged))
    print()

    if listing:
        for rel, line, text in hits:
            mark = ' ' if entry_key(rel, text) in allow else '*'
            print('%s %s:%d  %s' % (mark, rel, line, text))
        return 0

    if untriaged:
        print('FAIL: these can kill the process on NaN, infinity or an out-of-range value.')
        print('Guard with isFinite and clamp, or add to tools/trapconv_allowlist.txt')
        print('with a written reason if it is genuinely unreachable.')
        print()
        for rel, line, text in untriaged:
            # Printed in allowlist form so triaging one is a copy, a paste
            # and a reason, rather than a transcription with a chance to
            # get the line number wrong.
            print('  %s:%d' % (rel, line))
            print('    %s%s%s  # WHY IS THIS SAFE?' % (rel, ENTRY_SEP, text))
        return 1

    print('PASS: every trapping conversion has been looked at by a person.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
