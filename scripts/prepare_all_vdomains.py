#!/usr/bin/env python
"""
Merge & deduplicate all VH/VL variable domains into two FASTAs:
- Prefer large candidate sets if present:
    data/vh_candidates.fasta, data/vl_candidates.fasta
- Also include scFv-derived:
    data/vh.fasta, data/vl.fasta  (if present)
- Clean, length-filter, dedup, and write:
    data/all_vh.fasta
    data/all_vl.fasta
"""

import re, sys
from pathlib import Path
from textwrap import wrap

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

def read_fasta(path):
    out=[]; name=None; seq=[]
    if not Path(path).exists(): return out
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: out.append((name,"".join(seq)))
                name=line[1:].split()[0]
                seq=[]
            else:
                seq.append(line)
    if name: out.append((name,"".join(seq)))
    return out

def clean_seq(s):
    s = s.upper().replace(" ", "")
    s = re.sub(r"[^A-Z]", "", s)
    # keep only AA20 letters
    s = "".join([c for c in s if c in AA20])
    return s

def write_fasta(path, records):
    with open(path, "w") as out:
        for h, s in records:
            out.write(f">{h}\n")
            for chunk in wrap(s, 60):
                out.write(chunk + "\n")

def keep_v_length(s, lo=70, hi=170):
    return lo <= len(s) <= hi

def main():
    data = Path("data")
    inputs_vh = [
        data/"vh_candidates.fasta",
        data/"vh.fasta",           # scFv-derived (optional)
    ]
    inputs_vl = [
        data/"vl_candidates.fasta",
        data/"vl.fasta",           # scFv-derived (optional)
    ]

    raw_vh=[]; raw_vl=[]
    for p in inputs_vh: raw_vh += read_fasta(p)
    for p in inputs_vl: raw_vl += read_fasta(p)

    if not raw_vh and not raw_vl:
        print("No inputs found. Generate candidates first or check paths.")
        sys.exit(1)

    seen_vh=set(); seen_vl=set()
    vh_out=[]; vl_out=[]

    # keep first occurrence per unique sequence
    for h,s in raw_vh:
        cs = clean_seq(s)
        if not cs or not keep_v_length(cs): continue
        if cs in seen_vh: continue
        seen_vh.add(cs)
        vh_out.append((h, cs))

    for h,s in raw_vl:
        cs = clean_seq(s)
        if not cs or not keep_v_length(cs): continue
        if cs in seen_vl: continue
        seen_vl.add(cs)
        vl_out.append((h, cs))

    out_vh = data/"all_vh.fasta"
    out_vl = data/"all_vl.fasta"
    write_fasta(out_vh, vh_out)
    write_fasta(out_vl, vl_out)

    print(f"Inputs read: VH={len(raw_vh)} VL={len(raw_vl)}")
    print(f"Unique (length-filtered) -> VH={len(vh_out)} -> {out_vh}")
    print(f"Unique (length-filtered) -> VL={len(vl_out)} -> {out_vl}")
    print("Tip:")
    print("  grep -c '^>' data/all_vh.fasta")
    print("  grep -c '^>' data/all_vl.fasta")

if __name__ == "__main__":
    main()
