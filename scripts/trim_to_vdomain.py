import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide
from Bio.SeqUtils import seq1


# Defaults (override with CLI)
RAW_DIR_DEFAULT = "data/raw_pdb"
CSV_DEFAULT = "data/chains.csv"
OUT_DIR_DEFAULT = "data/processed/vdomains"

# Keep residue name mapping consistent with data_fil.py
CUSTOM_MAP_1L = {
    "MSE": "M",
    "FME": "M",
    "SEC": "C",
    "PYL": "K",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}


def chain_seq_and_index_map(chain) -> Tuple[str, List[Tuple[str, tuple]]]:
    """Return (sequence, index_map) for a PDB chain.

    sequence: single-letter AA string (non-AAs skipped)
    index_map: list of (chain_id, pdb_res_id) aligned to sequence indices
    """
    seq_chars: List[str] = []
    idx_map: List[Tuple[str, tuple]] = []
    for res in chain:
        if not Polypeptide.is_aa(res, standard=False):
            continue
        resname = res.get_resname().strip()
        try:
            aa = seq1(resname, custom_map=CUSTOM_MAP_1L)
        except Exception:
            continue
        if not aa or len(aa) != 1 or aa.upper() == "X":
            continue
        seq_chars.append(aa)
        idx_map.append((chain.id, res.get_id()))
    return "".join(seq_chars), idx_map


class VDomainSelect(Select):
    def __init__(self, allow: Dict[str, set], keep_hets: bool = False):
        self.allow = allow
        self.keep_hets = keep_hets

    def accept_chain(self, chain):
        return 1 if chain.id in self.allow else 0

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        if residue.id[0].strip():  # hetero/water
            return 1 if self.keep_hets and (chain_id in self.allow) else 0
        return 1 if (chain_id in self.allow and residue.id in self.allow[chain_id]) else 0


def parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def trim_one(pdb_path: str, row: dict, out_path: str, separate: bool = False, keep_hets: bool = False, debug: bool = False) -> List[str]:
    """Write trimmed PDB(s) for a single row. Returns list of written file paths."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())

    # Extract row fields
    pdb_id = row.get("pdb_id")
    vh_chain = (row.get("vh_chain") or "").strip()
    vl_chain = (row.get("vl_chain") or "").strip()
    vh_start = parse_int(row.get("vh_start") or "")
    vh_end = parse_int(row.get("vh_end") or "")
    vl_start = parse_int(row.get("vl_start") or "")
    vl_end = parse_int(row.get("vl_end") or "")
    vh_seq_csv = (row.get("vh_seq") or "").strip()
    vl_seq_csv = (row.get("vl_seq") or "").strip()

    # Build selection sets
    allow: Dict[str, set] = {}
    found = {"H": False, "L": False}

    chain_index: Dict[str, Tuple[str, List[Tuple[str, tuple]]]] = {}
    for chain in model:
        seq, idx_map = chain_seq_and_index_map(chain)
        chain_index[chain.id] = (seq, idx_map)

    # Helper to select residues for a domain
    def add_domain(chain_id: str, s_idx: Optional[int], e_idx: Optional[int], seq_csv: str, label: str):
        if not chain_id:
            return
        if chain_id not in chain_index:
            if debug:
                print(f"{pdb_id}: chain {chain_id} not found in PDB", file=sys.stderr)
            return
        seq, idx_map = chain_index[chain_id]
        if s_idx is None or e_idx is None or s_idx < 0 or e_idx >= len(seq) or s_idx > e_idx:
            # Try to locate by sequence string from CSV
            if seq_csv:
                pos = seq.find(seq_csv)
                if pos != -1:
                    s_loc = pos
                    e_loc = pos + len(seq_csv) - 1
                    if debug:
                        print(f"{pdb_id}: adjusted {label} indices by sequence match {s_loc}-{e_loc}", file=sys.stderr)
                else:
                    if debug:
                        print(f"{pdb_id}: cannot map {label} by indices or sequence", file=sys.stderr)
                    return
            else:
                if debug:
                    print(f"{pdb_id}: missing indices for {label}", file=sys.stderr)
                return
        else:
            s_loc, e_loc = s_idx, e_idx

        # Collect residue IDs
        sel = set()
        for i in range(max(0, s_loc), min(len(idx_map), e_loc + 1)):
            _cid, resid = idx_map[i]
            sel.add(resid)
        if not sel:
            if debug:
                print(f"{pdb_id}: empty selection for {label}", file=sys.stderr)
            return
        allow.setdefault(chain_id, set()).update(sel)
        found["H" if label == "VH" else "L"] = True

    # Add heavy and light if present
    add_domain(vh_chain, vh_start, vh_end, vh_seq_csv, "VH")
    add_domain(vl_chain, vl_start, vl_end, vl_seq_csv, "VL")

    written: List[str] = []
    if not allow:
        if debug:
            print(f"{pdb_id}: nothing to write (no VH/VL mapped)", file=sys.stderr)
        return written

    io = PDBIO()
    io.set_structure(structure)

    if separate:
        # Heavy
        if vh_chain and vh_chain in allow:
            path_h = os.path.join(os.path.dirname(out_path), f"{pdb_id}_VH_{vh_chain}.pdb")
            os.makedirs(os.path.dirname(path_h), exist_ok=True)
            io.save(path_h, select=VDomainSelect({vh_chain: allow[vh_chain]}, keep_hets=keep_hets))
            written.append(path_h)
        # Light
        if vl_chain and vl_chain in allow:
            path_l = os.path.join(os.path.dirname(out_path), f"{pdb_id}_VL_{vl_chain}.pdb")
            os.makedirs(os.path.dirname(path_l), exist_ok=True)
            io.save(path_l, select=VDomainSelect({vl_chain: allow[vl_chain]}, keep_hets=keep_hets))
            written.append(path_l)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        io.save(out_path, select=VDomainSelect(allow, keep_hets=keep_hets))
        written.append(out_path)

    return written


def main():
    ap = argparse.ArgumentParser(description="Trim PDBs to antibody V-domains based on data/chains.csv")
    ap.add_argument("--csv", default=CSV_DEFAULT, help="input CSV produced by data_fil.py")
    ap.add_argument("--raw_dir", default=RAW_DIR_DEFAULT, help="directory with source PDB files")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="output directory for trimmed PDBs")
    ap.add_argument("--only", nargs="*", default=None, help="optional PDB IDs to process")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--separate", action="store_true", help="write separate VH/VL PDBs instead of a combined file")
    ap.add_argument("--keep_hets", action="store_true", help="keep HETATM/waters for selected chains")
    ap.add_argument("--debug", action="store_true", help="verbose logging")
    ap.add_argument("--qc", action="store_true", help="verify trimmed sequences match CSV sequences")
    ap.add_argument("--qc_out", default=None, help="optional CSV to write QC results (mismatches only if provided)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    rows: List[dict] = []
    with open(args.csv, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if args.only:
        wanted = set(x.lower() for x in args.only)
        rows = [r for r in rows if (r.get("pdb_id") or "").lower() in wanted]

    if not rows:
        print("No rows to process.")
        return

    written_total = 0
    # Track outputs per row for QC
    outputs_per_row: List[Tuple[dict, List[str]]] = []
    for r in rows:
        pdb_id = r.get("pdb_id")
        if not pdb_id:
            continue
        in_path = os.path.join(args.raw_dir, f"{pdb_id}.pdb")
        if not os.path.exists(in_path):
            if args.debug:
                print(f"{pdb_id}: missing PDB {in_path}", file=sys.stderr)
            continue

        out_base = os.path.join(args.out_dir, f"{pdb_id}_vdomain.pdb")
        if (not args.overwrite) and (os.path.exists(out_base) or (args.separate and (os.path.exists(os.path.join(args.out_dir, f"{pdb_id}_VH_{r.get('vh_chain','')}.pdb") ) or os.path.exists(os.path.join(args.out_dir, f"{pdb_id}_VL_{r.get('vl_chain','')}.pdb") )))):
            if args.debug:
                print(f"{pdb_id}: outputs exist; skipping (use --overwrite)")
            continue

        paths = trim_one(in_path, r, out_base, separate=args.separate, keep_hets=args.keep_hets, debug=args.debug)
        if paths:
            written_total += len(paths)
            if args.debug:
                print(f"{pdb_id}: wrote {', '.join(paths)}")
        outputs_per_row.append((r, paths))

    print(f"Trimmed files written: {written_total}")

    # Optional QC step
    if args.qc:
        def extract_chain_seq(pdb_file: str, wanted_chain: str) -> str:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("qc", pdb_file)
            try:
                model = next(structure.get_models())
            except StopIteration:
                return ""
            for chain in model:
                if chain.id != wanted_chain:
                    continue
                seq, _ = chain_seq_and_index_map(chain)
                return seq
            return ""

        mismatches: List[List[str]] = []
        checked = 0
        ok = 0

        for r, paths in outputs_per_row:
            pdb_id = r.get("pdb_id") or ""
            vh_chain = (r.get("vh_chain") or "").strip()
            vl_chain = (r.get("vl_chain") or "").strip()
            vh_seq_csv = (r.get("vh_seq") or "").strip()
            vl_seq_csv = (r.get("vl_seq") or "").strip()

            # Determine which files to read
            files_to_check: List[Tuple[str, str, str]] = []  # (path, chain_id, label)
            if args.separate:
                if vh_chain:
                    files_to_check.append((os.path.join(args.out_dir, f"{pdb_id}_VH_{vh_chain}.pdb"), vh_chain, "VH"))
                if vl_chain:
                    files_to_check.append((os.path.join(args.out_dir, f"{pdb_id}_VL_{vl_chain}.pdb"), vl_chain, "VL"))
            else:
                comb = os.path.join(args.out_dir, f"{pdb_id}_vdomain.pdb")
                if vh_chain:
                    files_to_check.append((comb, vh_chain, "VH"))
                if vl_chain:
                    files_to_check.append((comb, vl_chain, "VL"))

            for path, chain_id, label in files_to_check:
                if not os.path.exists(path):
                    continue
                observed = extract_chain_seq(path, chain_id)
                expected = vh_seq_csv if label == "VH" else vl_seq_csv
                if not expected:
                    continue
                checked += 1
                if observed == expected:
                    ok += 1
                else:
                    mismatches.append([
                        pdb_id,
                        label,
                        chain_id,
                        str(len(expected)),
                        str(len(observed)),
                        expected,
                        observed,
                    ])

        # Report
        print(f"QC checked: {checked}, OK: {ok}, mismatches: {len(mismatches)}")
        if mismatches:
            if args.qc_out:
                os.makedirs(os.path.dirname(args.qc_out) or ".", exist_ok=True)
                with open(args.qc_out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["pdb_id", "domain", "chain", "expected_len", "observed_len", "expected_seq", "observed_seq"])
                    w.writerows(mismatches)
                print(f"QC report written: {args.qc_out}")
            else:
                # Print first few mismatches inline for visibility
                print("Examples of mismatches (first 3):")
                for row in mismatches[:3]:
                    print(f"- {row[0]} {row[1]} chain {row[2]}: expected_len={row[3]} observed_len={row[4]}")


if __name__ == "__main__":
    main()
