#!/usr/bin/env python3
"""Fix the local-variable-variant MAP(PRESENT) GPU-offload bug (see
../VASP_GPU_OMP_PRESENT_BUG_FIXES.txt) in VASP GPOMP directives: rewrites
MAP(PRESENT,TO:X) -> MAP(ALLOC:X) for any name X already registered via a
MAP(ALLOC:...)/MAP(ALWAYS,ALLOC:...) earlier in the SAME subroutine.
Paren-balance-aware, so it's safe on nested array-section expressions.
Modifies files in place. Run from the VASP src/ directory (no args), ideally
after a fresh detect_map_present_bug.py pass and with a backup on hand:
    python3 fix_map_present_bug.py

NOTE: does NOT fix the DUMMY-ARGUMENT variant (e.g. nonl.F's CACC/VNLACC
case) or multi-line (&-continued) MAP clauses - those still need manual
fixes (see the .txt report).
"""
import re, os, glob

files = sorted(fn for fn in glob.glob("*.F")
               if re.search(r'MAP\(\s*PRESENT', open(fn, errors="replace").read(), re.IGNORECASE))

sub_start_re = re.compile(r'^\s*(SUBROUTINE|FUNCTION)\s+(\w+)', re.IGNORECASE)
sub_end_re   = re.compile(r'^\s*END\s+(SUBROUTINE|FUNCTION)', re.IGNORECASE)
map_open_re  = re.compile(r'MAP\(', re.IGNORECASE)

def find_matching_paren(s, open_idx):
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1

def split_top_level(s):
    parts, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1; cur += ch
        elif ch == ')':
            depth -= 1; cur += ch
        elif ch == ',' and depth == 0:
            parts.append(cur); cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts

def base_name(n):
    return re.split(r'[(%]', n.strip())[0].strip().upper()

def process_line(line, cur_alloc_names):
    """Returns (new_line, num_directives_changed). Also mutates cur_alloc_names
    with any ALLOC targets seen (before AND after edits) on this line."""
    if not re.match(r'^\s*GPOMP\b', line, re.IGNORECASE):
        return line, 0

    spans = []  # (start_of_'M', open_paren_idx, close_paren_idx, kind, names)
    for m in map_open_re.finditer(line):
        open_idx = m.end() - 1
        close_idx = find_matching_paren(line, open_idx)
        if close_idx == -1:
            continue
        inner = line[open_idx+1:close_idx]
        stripped = inner.lstrip()
        if re.match(r'PRESENT\s*[,:]', stripped, re.IGNORECASE):
            colon = inner.find(':')
            names = split_top_level(inner[colon+1:]) if colon != -1 else []
            spans.append((m.start(), open_idx, close_idx, 'PRESENT', names))
        elif re.match(r'(ALWAYS\s*,\s*)?ALLOC\s*:', stripped, re.IGNORECASE):
            colon = inner.find(':')
            names = split_top_level(inner[colon+1:]) if colon != -1 else []
            spans.append((m.start(), open_idx, close_idx, 'ALLOC', names))
            for n in names:
                b = base_name(n)
                if b: cur_alloc_names.add(b)

    changed = 0
    # process rightmost-first so earlier indices stay valid during string edits
    for start, open_idx, close_idx, kind, names in sorted(spans, key=lambda t: -t[0]):
        if kind != 'PRESENT':
            continue
        keep, moved = [], []
        for n in names:
            b = base_name(n)
            (moved if b in cur_alloc_names else keep).append(n)
        if not moved:
            continue
        parts = []
        if keep:
            parts.append('MAP(PRESENT,TO:' + ','.join(keep) + ')')
        if moved:
            parts.append('MAP(ALLOC:' + ','.join(moved) + ')')
        replacement = ' '.join(parts)
        line = line[:start] + replacement + line[close_idx+1:]
        changed += 1

    return line, changed

total_changed = 0
for fn in files:
    if not os.path.exists(fn):
        print(f"{fn}: FILE NOT FOUND"); continue
    lines = open(fn, errors="replace").readlines()
    cur_alloc_names = set()
    changed = 0
    for i, line in enumerate(lines):
        if sub_start_re.match(line):
            cur_alloc_names = set()
        elif sub_end_re.match(line):
            cur_alloc_names = set()
        newline, n = process_line(line, cur_alloc_names)
        if n:
            lines[i] = newline
            changed += n
    if changed:
        open(fn, 'w').writelines(lines)
    print(f"{fn}: {changed} directive(s) modified")
    total_changed += changed

print(f"\nTOTAL directives modified: {total_changed}")
