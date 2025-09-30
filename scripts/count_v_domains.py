#!/usr/bin/env python
"""
Count candidate VH / VL variable domains directly from a nucleotide FASTA
(without requiring an scFv linker). Translates all 6 frames, finds AA segments
with IgV-like starts and typical V-domain length, and deduplicates.

Outputs:
- data/vh_candidates.fasta
- data/vl_candidates.fasta
Prints counts.
"""

import sys, re
from pathlib import Path
from textwrap import wrap
from Bio import SeqIO
from Bio.Seq import Seq

# Heuristics for chain starts (quick & practical)
VH_STARTS = ("EVQ", "QVQ", "QVQL", "QVV", "QVL", "EVI", "QVH", "QVR")
VL_STARTS = ("EIV", "DIV", "DIQ", "QIV", "QIQ", "QIVL", "QIS", "QISL", "EIVL")

def write_fasta(path, records):
    with open(path, "w") as out:
        for h, s in records:
            out.write(f">{h}\n")
            for chunk in wrap(s, 60):
                out.write(chunk + "\n")

def translate_all_frames(nt_seq):
    seq = Seq(nt_seq.upper().replace("U","T"))
    def translate_trimmed(s):
        n = len(s)
        if n % 3 != 0:
            s = s[: n - (n % 3)]
        return str(s.translate(to_stop=False))
    frames = []
    for f in range(3):
        frames.append(translate_trimmed(seq[f:]))
    rc = seq.reverse_complement()
    for f in range(3):
        frames.append(translate_trimmed(rc[f:]))
    return frames

def classify_v_chain(aa):
    head = aa[:10]
    if head.startswith(VH_STARTS): return "VH"
    if head.startswith(VL_STARTS): return "VL"
    return None

def main():
    in_fa = sys.argv[1] if len(sys.argv) > 1 else "data/imgt.fasta"
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True, parents=True)
    vh_out = out_dir / "vh_candidates.fasta"
    vl_out = out_dir / "vl_candidates.fasta"

    vh_set, vl_set = set(), set()
    vh_records, vl_records = [], []

    total_nt = 0
    for rec in SeqIO.parse(in_fa, "fasta"):
        total_nt += 1
        nt = re.sub(r"[^ACGTUacgtu]", "", str(rec.seq))
        if len(nt) < 240:  # too short for a V domain
            continue
        seen_this_record = set()
        for aa in translate_all_frames(nt):
            aa = aa.replace("*", "")
            if not (80 <= len(aa) <= 150):
                continue
            c = classify_v_chain(aa)
            if not c:
                continue
            # de-dup within record
            key = (c, aa)
            if key in seen_this_record:
                continue
            seen_this_record.add(key)
            # global de-dup
            if c == "VH" and aa not in vh_set:
                vh_set.add(aa)
                vh_records.append((f"{rec.id.replace('|','_')}_VH", aa))
            elif c == "VL" and aa not in vl_set:
                vl_set.add(aa)
                vl_records.append((f"{rec.id.replace('|','_')}_VL", aa))

    write_fasta(vh_out, vh_records)
    write_fasta(vl_out, vl_records)

    print(f"Parsed nucleotide records: {total_nt}")
    print(f"Unique VH candidates: {len(vh_set)} -> {vh_out}")
    print(f"Unique VL candidates: {len(vl_set)} -> {vl_out}")
    print("Tip: grep -c '^>' on the outputs to double-check counts.")

if __name__ == "__main__":
    main()
