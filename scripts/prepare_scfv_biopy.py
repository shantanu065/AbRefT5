#!/usr/bin/env python
"""
Extract scFv-like sequences from IMGT nucleotide FASTA:
- Translate all 6 frames with Biopython
- Detect scFv via (GGGGS){n} linker
- Split into VH / VL
- Write: data/scfv_aa.fasta, data/vh.fasta, data/vl.fasta
- Create simple train/val splits
"""

import re, sys, csv, random
from pathlib import Path
from textwrap import wrap
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

# scFv linker (G4S repeats)
AA_LINKER_RE = re.compile(r"(GGGGS){2,6}")

# Simple VH/VL start heuristics
VH_STARTS = ("EVQ", "QVQ", "QVL", "QVV", "EVI", "QVH", "QVR")
VL_STARTS = ("EIV", "DIV", "DIQ", "QIV", "QIQ", "QIVL", "QIS")

def write_fasta(path, records):
    with open(path, "w") as out:
        for h, s in records:
            out.write(f">{h}\n")
            for chunk in wrap(s, 60):
                out.write(chunk + "\n")

def split_on_linker(sc):
    m = AA_LINKER_RE.search(sc)
    if not m:
        return None, None
    i, j = m.span()
    left, right = sc[:i], sc[j:]
    # try assign by starts
    if left.startswith(VH_STARTS) and right.startswith(VL_STARTS):
        return left, right
    if left.startswith(VL_STARTS) and right.startswith(VH_STARTS):
        return right, left
    return left, right  # fallback

def looks_like_igV(seq):
    head = seq[:10]
    return head.startswith(VH_STARTS) or head.startswith(VL_STARTS)

def translate_all_frames(nt_seq):
    from Bio.Seq import Seq
    seq = Seq(nt_seq.upper().replace("U","T"))
    frames = []

    def translate_trimmed(s):
        n = len(s)
        if n % 3 != 0:
            s = s[: n - (n % 3)]  # trim trailing bases so len % 3 == 0
        return str(s.translate(to_stop=False))

    # forward frames
    for f in range(3):
        frames.append(translate_trimmed(seq[f:]))

    # reverse-complement frames
    rc = seq.reverse_complement()
    for f in range(3):
        frames.append(translate_trimmed(rc[f:]))

    return frames

def main():
    # Input FASTA (nucleotide, from IMGT)
    in_fasta = sys.argv[1] if len(sys.argv) > 1 else "data/imgt.fasta"

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True, parents=True)
    scfv_out = out_dir / "scfv_aa.fasta"
    vh_out   = out_dir / "vh.fasta"
    vl_out   = out_dir / "vl.fasta"
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    scfv_records, vh_records, vl_records = [], [], []

    records = list(SeqIO.parse(in_fasta, "fasta"))
    for rec in tqdm(records, desc="Translating & filtering", unit="seq"):
        nt = str(rec.seq)
        nt = re.sub(r"[^ACGTUacgtu]", "", nt)
        if len(nt) < 450:  # scFv typically > ~600 nt; keep generous cutoff
            continue
        for aa in translate_all_frames(nt):
            aa = aa.replace("*", "")  # drop stops crudely
            if not (200 <= len(aa) <= 320):
                continue
            if AA_LINKER_RE.search(aa) and looks_like_igV(aa):
                vh, vl = split_on_linker(aa)
                if vh and vl and (80 <= len(vh) <= 140) and (80 <= len(vl) <= 140):
                    hid = rec.id.replace("|","_")
                    scfv_records.append((hid, aa))
                    vh_records.append((hid + "_VH", vh))
                    vl_records.append((hid + "_VL", vl))
                    break

    if not scfv_records:
        print("No scFv sequences detected - adjust heuristics or input.")
        return

    write_fasta(scfv_out, scfv_records)
    write_fasta(vh_out, vh_records)
    write_fasta(vl_out, vl_records)

    # simple 90/10 split by scFv ids
    ids = [h for h,_ in scfv_records]
    random.seed(42); random.shuffle(ids)
    k = max(1, int(0.1 * len(ids)))

    with open(splits_dir/"train.txt","w") as f:
        for i in ids[k:]:
            f.write(i+"\n")
    with open(splits_dir/"val.txt","w") as f:
        for i in ids[:k]:
            f.write(i+"\n")

    print(f"scFv={len(scfv_records)}, VH={len(vh_records)}, VL={len(vl_records)}")
    print(f"Wrote: {scfv_out}, {vh_out}, {vl_out}")
    print(f"Splits in: {splits_dir}")

if __name__ == "__main__":
    main()
