#!/usr/bin/env python
"""
Create refinement pairs (noisy -> clean) from all_vh/all_vl for seq2seq training.

Inputs (defaults):
  data/all_vh.fasta
  data/all_vl.fasta

Outputs:
  data/train_refine.tsv   (ctrl \t noisy \t target)
  data/val_refine.tsv     (ctrl \t noisy \t target)
"""

import os, sys, csv, re, random
from pathlib import Path

random.seed(42)
AA = "ACDEFGHIKLMNPQRSTVWY"

def read_fasta(path):
    out=[]; name=None; seq=[]
    if not Path(path).exists(): return out
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: out.append((name,"".join(seq).upper().replace(" ", "")))
                name=line[1:].split()[0]
                seq=[]
            else:
                seq.append(line)
    if name: out.append((name,"".join(seq).upper().replace(" ", "")))
    return out

def rand_aa(n): 
    return "".join(random.choice(AA) for _ in range(n))

def jitter(seq):
    s = list(seq)

    # random swaps
    for _ in range(max(1, len(s)//25)):
        i = random.randrange(len(s)); j = random.randrange(len(s))
        s[i], s[j] = s[j], s[i]
    out = "".join(s)

    # insertion
    if random.random() < 0.5:
        pos = random.randrange(0, len(out)+1)
        out = out[:pos] + rand_aa(random.randint(5, 15)) + out[pos:]

    # deletion
    if random.random() < 0.5 and len(out) > 40:
        i = random.randrange(0, len(out)-10)
        out = out[:i] + out[i+random.randint(3, 8):]

    # duplication
    if random.random() < 0.5 and len(out) > 40:
        i = random.randrange(0, len(out)-10)
        span = out[i:i+random.randint(3, 8)]
        out = out[:i] + span + out[i:]

    # homopolymer run
    if random.random() < 0.4:
        aa = random.choice(AA)
        pos = random.randrange(0, len(out)+1)
        out = out[:pos] + aa * random.randint(5, 10) + out[pos:]

    # small subs
    if random.random() < 0.7:
        s = list(out)
        for _ in range(max(1, len(s)//30)):
            k = random.randrange(len(s))
            s[k] = random.choice(AA)
        out = "".join(s)

    out = re.sub(r"[^A-Z]", "", out)[:512]
    return out

def main():
    vh_path = sys.argv[1] if len(sys.argv) > 1 else "data/all_vh.fasta"
    vl_path = sys.argv[2] if len(sys.argv) > 2 else "data/all_vl.fasta"

    vh = read_fasta(vh_path)
    vl = read_fasta(vl_path)
    if not vh and not vl:
        print("No input sequences found. Run prepare_all_vdomains first.")
        sys.exit(1)

    out_dir = Path("data"); out_dir.mkdir(exist_ok=True, parents=True)
    train_tsv = out_dir/"train_refine.tsv"
    val_tsv   = out_dir/"val_refine.tsv"

    rows=[]
    def add(recs, chain):
        for _, tgt in recs:
            if not (70 <= len(tgt) <= 170):  # Ig V-domain guard
                continue
            for _ in range(3):               # 3 noisy variants per clean target
                noisy = jitter(tgt)
                rows.append([f"[REFINE] chain={chain}", noisy, tgt])

    add(vh, "VH")
    add(vl, "VL")

    random.shuffle(rows)
    n=len(rows); cut=max(1, int(0.9*n))
    with open(train_tsv, "w", newline="") as f:
        w=csv.writer(f, delimiter="\t"); w.writerows(rows[:cut])
    with open(val_tsv, "w", newline="") as f:
        w=csv.writer(f, delimiter="\t"); w.writerows(rows[cut:])

    print(f"Built refine pairs: total={n}  train={cut}  val={n-cut}")
    print(f"Wrote: {train_tsv}")
    print(f"Wrote: {val_tsv}")
    if rows:
        print("Preview:")
        ctrl,noisy,tgt = rows[0]
        print("\t".join([ctrl, noisy[:50]+("..." if len(noisy)>50 else ""), tgt[:50]+("..." if len(tgt)>50 else "")]))

if __name__ == "__main__":
    main()
