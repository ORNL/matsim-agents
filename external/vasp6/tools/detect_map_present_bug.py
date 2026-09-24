#!/usr/bin/env python3
"""Detect the local-variable-variant MAP(PRESENT) GPU-offload bug (see
../VASP_GPU_OMP_PRESENT_BUG_FIXES.txt) in VASP GPOMP directives: for each
subroutine, flags any name that appears in a MAP(PRESENT,TO:...) clause
after its own MAP(ALLOC:...)/MAP(ALWAYS,ALLOC:...) registration earlier in
that same subroutine. Read-only (no files are modified). Run from the VASP
src/ directory: python3 detect_map_present_bug.py

NOTE: this only catches the LOCAL-VARIABLE variant. It cannot catch the
DUMMY-ARGUMENT variant (e.g. nonl.F's CACC/VNLACC case), which requires
manual call-chain tracing, nor multi-line (&-continued) MAP clauses.
"""
import re, os, glob

files = sorted(fn for fn in glob.glob("*.F")
               if re.search(r'MAP\(\s*PRESENT', open(fn, errors="replace").read(), re.IGNORECASE))

sub_start_re = re.compile(r'^\s*(SUBROUTINE|FUNCTION)\s+(\w+)', re.IGNORECASE)
sub_end_re   = re.compile(r'^\s*END\s+(SUBROUTINE|FUNCTION)', re.IGNORECASE)
alloc_re     = re.compile(r'MAP\(\s*(?:ALWAYS,\s*)?ALLOC\s*:\s*([^)]+)\)', re.IGNORECASE)
present_re   = re.compile(r'MAP\(\s*PRESENT\s*,\s*TO\s*:\s*([^)]+)\)', re.IGNORECASE)
total=0
for fn in files:
    lines = open(fn, errors="replace").readlines()
    cur=set(); flagged=[]
    for i, line in enumerate(lines,1):
        if sub_start_re.match(line): cur=set()
        elif sub_end_re.match(line): cur=set()
        if not re.match(r'^\s*GPOMP\b', line, re.IGNORECASE): continue
        for m in alloc_re.finditer(line):
            for n in m.group(1).split(','):
                b=re.split(r'[(%]', n.strip())[0].strip()
                if b: cur.add(b.upper())
        for m in present_re.finditer(line):
            for n in m.group(1).split(','):
                b=re.split(r'[(%]', n.strip())[0].strip().upper()
                if b in cur: flagged.append((i,b,line.strip()))
    print(f"{fn}: {len(flagged)} flagged")
    total+=len(flagged)
print(f"TOTAL: {total}")
