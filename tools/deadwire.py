r"""Find code that was WRITTEN AND NEVER WIRED.

Four instances of this have already cost this project dearly, each one found by
accident rather than by looking:

  pruneStartFraction/pruneEndFraction  declared, defaulted, assigned, read nowhere
                                       -> pruning ran ungated, 29 passes not 10
  seedTarget(forSplatCap:)             documented with worked examples, called nowhere
                                       -> densification had zero headroom
  hasNormal (PrePassICP)               declared, never written or read
                                       -> loop closure never ran, in any scan
  SplatCloud.fuse3DFilter(_:)          written, correct, called nowhere
                                       -> every export drawn at wrong opacity

All four COMPILED. All four looked configured. None of them ran.

This finds the rest. It reports a declaration whose name appears nowhere in the
module except its own declaration, its doc comment, and (for properties) its own
initialiser. Swift is one module here, so "nothing references this name" really
does mean dead, with the exceptions listed in SKIP below.

--------------------------------------------------------------------------------
RUNNING IT IN CI
--------------------------------------------------------------------------------
    python tools/deadwire.py                # exit 0 if nothing NEW is dead
    python tools/deadwire.py --list         # print every dead name, exit 0
    python tools/deadwire.py --list-new     # print only unallowed ones, exit 0

Every candidate that a human has looked at and judged harmless lives in

    tools/deadwire_allowlist.txt

one `File.swift:name  # reason` per line. Those are excluded. ANYTHING ELSE
fails the build.

That asymmetry is the whole point. Failing on the ~110 known-harmless
candidates would be a check that is always red, and a check that is always red
is a check everybody learns to scroll past. Failing ONLY on a name that nobody
has triaged means the build goes red exactly when somebody writes new code that
nothing calls, which is the shape of all four bugs above.

If this fails on your branch you have two honest options and no third one:

  * WIRE IT UP. This is the right answer whenever the declaration implements
    something the app is supposed to do. Deleting a correct implementation of a
    real feature makes the app worse while making this check green.
  * If it really is a spare helper or a scanner false positive, add it to the
    allowlist WITH A REASON. The reason is the part that matters: it is the
    record that a person looked, and it is what the next person reads instead
    of re-deriving the answer.

--------------------------------------------------------------------------------
TWO FAMILIES OF FALSE POSITIVE, both already reflected in the allowlist
--------------------------------------------------------------------------------
1. Locals used only inside a string interpolation. `let freeMB = ...` then
   `"\(freeMB) MB free"`. Strings are stripped before counting, so the use
   disappears and the name looks dead. Very common in the census and logging.
2. Swift mirrors of Metal structs (`rot0`, `mean2DX`, `conicA`, `ssimC1` in
   TrainerGPULayouts.swift and ViewerGPULayouts.swift). These exist to fix the
   memory layout the GPU reads. Swift never names them and never should.
"""
import io
import os
import re
import sys
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Portable: derived from this file's own location, so it works on the macOS CI
# runner and on a Windows checkout alike. Override with argv for a one-off.
DEFAULT_ROOT = os.path.join(REPO, 'ios', 'Sources')
ALLOWLIST = os.path.join(HERE, 'deadwire_allowlist.txt')

# Names that are legitimately never referenced by our own code.
SKIP_EXACT = {
    'body', 'main', 'init', 'deinit', 'description', 'errorDescription',
    'id', 'hashValue', 'rawValue', 'allCases', 'makeUIView', 'updateUIView',
    'makeCoordinator', 'encode', 'shared', 'default', 'zero', 'identity',
    'localizedDescription', 'debugDescription', 'placeholder',
}
# Prefixes/suffixes that indicate a protocol requirement or framework callback.
SKIP_PREFIX = ('session', 'renderer', 'application', 'scene', 'view', 'draw',
               'mtkView', 'observeValue', 'urlSession', 'connection', 'browser')

decl_patterns = [
    # func name(   /  static func name(
    (re.compile(r'^\s*(?:@\w+\s+)*(?:public\s+|internal\s+|private\s+|fileprivate\s+|open\s+)?'
                r'(?:static\s+|class\s+|final\s+|mutating\s+|nonisolated\s+)*'
                r'func\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]'), 'func'),
    # var/let name: Type      (stored or computed)
    (re.compile(r'^\s*(?:@\w+\s+)*(?:public\s+|internal\s+|private\s+|fileprivate\s+|open\s+)?'
                r'(?:static\s+|class\s+|final\s+|lazy\s+)*'
                r'(?:var|let)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:=]'), 'prop'),
]


def strip_comments_and_strings(text):
    out = []
    i, n = 0, len(text)
    in_line, in_block, in_str = False, 0, False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ''
        if in_line:
            if c == '\n':
                in_line = False
                out.append(c)
            else:
                out.append(' ')
            i += 1
        elif in_block:
            if c == '*' and nxt == '/':
                in_block -= 1
                out.append('  ')
                i += 2
            elif c == '/' and nxt == '*':
                in_block += 1
                out.append('  ')
                i += 2
            else:
                out.append('\n' if c == '\n' else ' ')
                i += 1
        elif in_str:
            if c == '\\':
                out.append('  ')
                i += 2
            elif c == '"':
                in_str = False
                out.append(' ')
                i += 1
            else:
                out.append('\n' if c == '\n' else ' ')
                i += 1
        else:
            if c == '/' and nxt == '/':
                in_line = True
                out.append('  ')
                i += 2
            elif c == '/' and nxt == '*':
                in_block = 1
                out.append('  ')
                i += 2
            elif c == '"':
                in_str = True
                out.append(' ')
                i += 1
            else:
                out.append(c)
                i += 1
    return ''.join(out)


def load_allowlist(path):
    """`{'File.swift:name': 'reason'}`. A missing file is an empty allowlist,
    which fails loudly on everything - the right way round for a guard."""
    allowed = {}
    if not os.path.exists(path):
        return allowed
    for raw in io.open(path, encoding='utf-8'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, _, reason = line.partition('#')
        key = key.strip()
        if key:
            allowed[key] = reason.strip()
    return allowed


def find_dead(root):
    files = []
    for dirpath, _, names in os.walk(root):
        for fn in names:
            if fn.endswith('.swift'):
                files.append(os.path.join(dirpath, fn))

    raw, code = {}, {}
    for p in files:
        t = io.open(p, encoding='utf-8', errors='replace').read()
        raw[p] = t
        code[p] = strip_comments_and_strings(t)

    # Every identifier occurrence in CODE (comments and strings removed).
    ident = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
    counts = collections.Counter()
    for p in files:
        for m in ident.finditer(code[p]):
            counts[m.group(0)] += 1

    decls = []
    for p in files:
        lines = code[p].split('\n')
        rawlines = raw[p].split('\n')
        for i, ln in enumerate(lines):
            for pat, kind in decl_patterns:
                m = pat.match(ln)
                if m:
                    decls.append((p, i + 1, m.group(1), kind, rawlines[i].strip()[:110]))
                    break

    dead = []
    for p, line, name, kind, text in decls:
        if name in SKIP_EXACT:
            continue
        if any(name.startswith(x) for x in SKIP_PREFIX):
            continue
        if name.startswith('_'):
            continue
        # One occurrence == only the declaration itself.
        if counts[name] <= 1:
            rel = os.path.relpath(p, root).replace('\\', '/')
            dead.append((rel, line, name, kind, text))

    dead.sort(key=lambda d: (d[0], d[1]))
    return dead


def key_for(rel_path, name):
    """`Capture/CaptureTuning.swift` + `foo` -> `CaptureTuning.swift:foo`.

    Basename rather than the full path, because the module directories are the
    one thing here that gets reorganised, and every .swift basename under
    Sources/ is unique (checked: 119 files, no collisions)."""
    return '%s:%s' % (os.path.basename(rel_path), name)


def main(argv):
    mode = argv[1] if len(argv) > 1 and argv[1].startswith('--') else None
    root = argv[2] if mode and len(argv) > 2 else (argv[1] if len(argv) > 1 and not mode else DEFAULT_ROOT)
    root = os.path.abspath(root)

    if not os.path.isdir(root):
        print('deadwire: no such directory: %s' % root)
        return 2

    dead = find_dead(root)
    allowed = load_allowlist(ALLOWLIST)

    triaged, untriaged = [], []
    for entry in dead:
        (rel, line, name, kind, text) = entry
        if key_for(rel, name) in allowed:
            triaged.append(entry)
        else:
            untriaged.append(entry)

    if mode == '--list':
        for rel, line, name, kind, text in dead:
            print(key_for(rel, name))
        return 0
    if mode == '--list-new':
        for rel, line, name, kind, text in untriaged:
            print(key_for(rel, name))
        return 0

    print('deadwire: scanning %s' % root)
    print('  declarations whose name appears NOWHERE else in the module: %d' % len(dead))
    print('  triaged and allowlisted (a person looked, and wrote down why): %d' % len(triaged))
    print('  NOT triaged: %d' % len(untriaged))
    print('')

    # An allowlist entry for something that is no longer dead is not a failure -
    # somebody wired it up, which is the outcome this whole check exists to
    # produce. It is reported so the list does not silently rot.
    still_dead = set(key_for(r, n) for r, _l, n, _k, _t in dead)
    stale = sorted(k for k in allowed if k not in still_dead)
    if stale:
        print('  %d allowlist entr%s no longer dead (wired up since, or renamed).'
              % (len(stale), 'y is' if len(stale) == 1 else 'ies are'))
        print('  Delete these lines from tools/deadwire_allowlist.txt:')
        for k in stale:
            print('      %s' % k)
        print('')

    if not untriaged:
        print('PASS: nothing dead that a person has not already looked at.')
        return 0

    by_file = collections.defaultdict(list)
    for rel, line, name, kind, text in untriaged:
        by_file[rel].append((line, name, kind, text))

    print('FAIL: %d declaration(s) that nothing calls, and that nobody has triaged.' % len(untriaged))
    print('')
    print('Each of these COMPILES and none of them RUNS. Ask of each one:')
    print('  "if this is never called, what feature silently does not happen?"')
    print('If the answer names a feature, WIRE IT UP.')
    print('If the answer is "nothing, it is a spare helper" or it is one of the')
    print('two false-positive families in this file\'s header, add it to')
    print('tools/deadwire_allowlist.txt with a one-line reason.')
    print('')
    for f in sorted(by_file):
        print('  %s' % f)
        for line, name, kind, text in by_file[f]:
            print('     %5d  %-4s %-34s %s' % (line, kind, name, text))
        print('')
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
