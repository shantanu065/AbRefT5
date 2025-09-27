import os, csv, sys, argparse
from typing import List, Tuple, Optional
from Bio.PDB import PDBParser, Polypeptide
from Bio.SeqUtils import seq1
from anarci import run_anarci

# Defaults (can be overridden by CLI)
RAW_DIR_DEFAULT = "data/raw_pdb"
OUT_CSV_DEFAULT = "data/chains.csv"
SCHEME_DEFAULT  = "chothia"  # or "imgt"

# Map common nonstandard residues to canonical one-letter codes
# Prefer mapping to the 20 canonical letters so ANARCI is happy.
CUSTOM_MAP_1L = {
    "MSE": "M",  # selenomethionine
    "FME": "M",
    "SEC": "C",  # map selenocysteine -> C (ANARCI expects 20 AAs)
    "PYL": "K",  # map pyrrolysine -> K
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}

def chain_seq_and_index_map(chain) -> Tuple[str, List[Tuple[int, str, tuple]]]:
    """
    Return a one-letter AA sequence for a chain and a map from sequence index
    to PDB residue identifiers: (model_idx(0), chain_id, res.get_id()).
    Only standard amino acids are included (non-AA and unmapped residues skipped).
    """
    seq_chars = []
    idx_map = []
    for res in chain:
        if not Polypeptide.is_aa(res, standard=False):
            continue
        resname = res.get_resname().strip()
        try:
            aa = seq1(resname, custom_map=CUSTOM_MAP_1L)
        except Exception:
            continue
        # Keep only valid single-letter AAs
        if not aa or len(aa) != 1 or aa.upper() == "X":
            continue
        seq_chars.append(aa)
        idx_map.append((0, chain.id, res.get_id()))  # model 0 assumed
    return "".join(seq_chars), idx_map

def _extract_anarci_domains(res_obj, debug: bool=False):
    """
    Normalize ANARCI output into a list of domain dicts with keys:
      { 'chain_type': 'H'|'K'|'L', 'numbering': list, 'start': int|None, 'end': int|None }
    Returns [] if nothing could be parsed.
    """
    domains = []
    try:
        # Newer ANARCI returns a tuple: (input_labels, results)
        if isinstance(res_obj, tuple) and len(res_obj) >= 2:
            results = res_obj[1]
        else:
            results = res_obj[0] if (isinstance(res_obj, (list, tuple)) and len(res_obj) >= 1) else res_obj
        if not results:
            return []
        seq_hits = results[0]
        if not seq_hits:
            return []
        # Handle multiple possible shapes
        # 1) [('H', [dom, dom2]), ('K', [dom]), ...]
        if isinstance(seq_hits, (list, tuple)) and seq_hits and all(isinstance(x, (list, tuple)) for x in seq_hits):
            for hit in seq_hits:
                if len(hit) < 2:
                    continue
                chain_type = hit[0]
                details = hit[1]
                if isinstance(details, dict):
                    doms = [details]
                elif isinstance(details, (list, tuple)):
                    doms = [d for d in details if isinstance(d, dict)]
                else:
                    doms = []
                for d in doms:
                    numbering = d.get("numbering") if isinstance(d, dict) else None
                    if numbering is None:
                        continue
                    domains.append({
                        "chain_type": chain_type,
                        "numbering": numbering,
                        "start": d.get("start"),
                        "end": d.get("end"),
                    })
        # 2) [dom_dict, dom_dict2, ...]
        if not domains and isinstance(seq_hits, (list, tuple)) and seq_hits and all(isinstance(d, dict) for d in seq_hits):
            for d in seq_hits:
                numbering = d.get("numbering")
                ctype = d.get("chain_type", None)
                if numbering is None:
                    continue
                domains.append({
                    "chain_type": ctype,
                    "numbering": numbering,
                    "start": d.get("start"),
                    "end": d.get("end"),
                })
        # 3) Single dict
        if not domains and isinstance(seq_hits, dict):
            numbering = seq_hits.get("numbering")
            ctype = seq_hits.get("chain_type", None)
            if numbering is not None:
                domains.append({
                    "chain_type": ctype,
                    "numbering": numbering,
                    "start": seq_hits.get("start"),
                    "end": seq_hits.get("end"),
                })
        # 4) Raw numbering table: a list/tuple of domains, where each domain is either
        #    a list of ((num, insert), aa) pairs OR a tuple whose first element is that list.
        if not domains and isinstance(seq_hits, (list, tuple)) and seq_hits:
            for dom in seq_hits:
                cand = None
                if isinstance(dom, (list, tuple)) and dom and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in dom):
                    cand = dom
                elif isinstance(dom, (list, tuple)) and len(dom) >= 1 and isinstance(dom[0], (list, tuple)):
                    inner = dom[0]
                    if inner and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in inner):
                        cand = inner
                if cand:
                    letters = ''.join(aa for (_pos, aa) in cand if isinstance(aa, str) and aa != '-')
                    numbering = [(_pos, aa) for (_pos, aa) in cand]
                    domains.append({
                        "chain_type": None,
                        "numbering": numbering,
                        "letters": letters,
                        "start": None,
                        "end": None,
                    })
    except Exception as e:
        if debug:
            print(f"ANARCI parse error: {e}", file=sys.stderr)
    return domains


def _guess_chain_type(domain_seq: str, chain_id: Optional[str] = None) -> Optional[str]:
    s = domain_seq.upper()
    if chain_id:
        if chain_id.upper() == 'H':
            return 'H'
        if chain_id.upper() == 'L':
            return 'L'
    heavy_prefixes = (
        'QVQL', 'EVQL', 'QVQ', 'EVQ', 'QVEL', 'QLQL', 'QVTL', 'EVHL', 'QVHL'
    )
    light_prefixes = (
        'ELVMTQ', 'DIVMTQ', 'DIQMTQ', 'EIVLTQ', 'QSVLTQ', 'QLVLTQ', 'LVMTQ',
        'AIQLTQ', 'KLVMTQ', 'QIVLTQ', 'EVVLTQ', 'QSVLT', 'QPVLTQ'
    )
    if s.startswith(heavy_prefixes):
        return 'H'
    if s.startswith(light_prefixes):
        return 'L'
    if 'FGGGTK' in s or s.endswith('FGGGTKL'):
        return 'L'
    if 'DYW' in s or 'WGQ' in s or 'WGAG' in s:
        return 'H'
    return None


def anarci_variable_span(
    seq: str,
    scheme: str = "chothia",
    debug: bool = False,
    ncpu: int = 1,
    species: Optional[List[str]] = None,
    dump_raw: bool = False,
) -> Optional[Tuple[str, int, int]]:
    """
    Run ANARCI on the full chain sequence.
    Returns (chain_type, start_idx, end_idx_inclusive) for the VARIABLE domain span in the sequence.
    chain_type is 'H' for heavy, 'K' or 'L' for light (depending on ANARCI).
    """
    try:
        kwargs = {"scheme": scheme, "ncpu": max(1, int(ncpu))}
        if species:
            kwargs["allowed_species"] = species
        res = run_anarci([("chain", seq)], **kwargs)
        if dump_raw and debug:
            try:
                preview = str(res)
                if len(preview) > 1200:
                    preview = preview[:1200] + " ... <truncated>"
                print("ANARCI raw result preview:\n" + preview, file=sys.stderr)
            except Exception:
                pass
        doms = _extract_anarci_domains(res, debug=debug)
        if not doms:
            if debug:
                print("ANARCI: no domains detected for sequence", file=sys.stderr)
            return None
        # Select the best domain: prefer the longest by letter count
        best_dom = None
        best_len = -1
        for d in doms:
            letters = d.get('letters')
            if not letters and isinstance(d.get('numbering'), list):
                # reconstruct letters if only numbering present
                try:
                    letters = ''.join(aa for (_pos, aa) in d['numbering'] if isinstance(aa, str) and aa != '-')
                    d['letters'] = letters
                except Exception:
                    letters = None
            if not letters:
                continue
            L = len(letters)
            if L > best_len:
                best_len = L
                best_dom = d
        if not best_dom:
            return None
        letters = best_dom['letters']
        chain_type = best_dom.get('chain_type')
        if not chain_type:
            chain_type = _guess_chain_type(letters)
    except Exception as e:
        if debug:
            print(f"ANARCI failed: {e}", file=sys.stderr)
        return None

    start = None
    end = None
    # Map the domain letters back to the chain sequence to get indices
    start = seq.find(letters)
    if start == -1:
        # Fallback: try a looser match by stripping potential gaps (already stripped)
        return None
    end = start + len(letters) - 1
    if start is None or end is None:
        return None
    return chain_type, start, end

def process_pdb(
    pdb_path: str,
    scheme: str,
    min_len: int,
    debug: bool = False,
    ncpu: int = 1,
    species: Optional[List[str]] = None,
    accept_heavy_only: bool = False,
    accept_light_only: bool = False,
    dump_raw: bool = False,
    nterm_max: int = 0,
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())  # use first model
    vh_candidates = []
    vl_candidates = []

    for chain in model:
        seq, idx_map = chain_seq_and_index_map(chain)
        if debug:
            print(f"PDB {os.path.basename(pdb_path)} chain {chain.id}: seq_len={len(seq)}", file=sys.stderr)
        if len(seq) < min_len:  # skip tiny chains quickly
            continue
        span = anarci_variable_span(seq, scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)
        if span is None:
            # Retry on N-terminus only if requested
            if nterm_max and nterm_max > 0:
                if debug:
                    print(f"ANARCI miss on full chain {chain.id}; trying N-terminus clip {nterm_max}", file=sys.stderr)
                seq_clip = seq[:nterm_max]
                span = anarci_variable_span(seq_clip, scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)
                if span is None:
                    continue
            else:
                # No fallback configured and no span -> skip
                continue
        ctype, s, e = span  # indices in seq (0-based, inclusive)
        length = e - s + 1
        vseq = seq[s:e+1]
        if ctype == "H":
            vh_candidates.append((chain.id, s, e, length, vseq))
        else:  # 'K' or 'L'
            vl_candidates.append((chain.id, s, e, length, vseq))

    # Pick the longest VH and VL variable domains if multiple candidates exist
    vh_candidates.sort(key=lambda x: x[3], reverse=True)
    vl_candidates.sort(key=lambda x: x[3], reverse=True)

    vh = vh_candidates[0] if vh_candidates else None
    vl = vl_candidates[0] if vl_candidates else None
    # Apply acceptance policy
    if vh and vl:
        return vh, vl
    if vh and not vl and accept_heavy_only:
        return vh, None
    if vl and not vh and accept_light_only:
        return None, vl
    return vh if vh else None, vl if vl else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process first N PDBs")
    ap.add_argument("--scheme", choices=["chothia", "imgt", "kabat", "aho"], default=SCHEME_DEFAULT, help="ANARCI numbering scheme")
    ap.add_argument("--raw_dir", default=RAW_DIR_DEFAULT, help="directory containing input .pdb files")
    ap.add_argument("--out_csv", default=OUT_CSV_DEFAULT, help="output CSV path")
    ap.add_argument("--min_len", type=int, default=50, help="minimum chain length to consider")
    ap.add_argument("--debug", action="store_true", help="verbose diagnostics for skipped chains")
    ap.add_argument("--ncpu", type=int, default=(os.cpu_count() or 1), help="threads to use inside ANARCI/HMMER")
    ap.add_argument("--species", nargs="*", default=None, help="restrict ANARCI to species (e.g. human mouse)")
    ap.add_argument("--resume", action="store_true", help="skip PDB IDs already present in out_csv")
    ap.add_argument("--append", action="store_true", help="append to out_csv instead of overwriting")
    ap.add_argument("--accept_heavy_only", action="store_true", help="write rows for heavy-only (nanobody/VHH) entries")
    ap.add_argument("--accept_light_only", action="store_true", help="write rows for light-only entries")
    ap.add_argument("--dump_anarci", action="store_true", help="print raw ANARCI result preview (debug)")
    ap.add_argument("--only", nargs="*", default=None, help="optional list of specific PDB IDs to process (without .pdb)")
    ap.add_argument("--nterm_max", type=int, default=0, help="if >0, retry ANARCI on only the first N residues when full-length fails")
    ap.add_argument("--show_seq", action="store_true", help="print extracted VH/VL sequences in the console")
    ap.add_argument("--out_fasta_vh", default=None, help="optional output FASTA path for VH sequences")
    ap.add_argument("--out_fasta_vl", default=None, help="optional output FASTA path for VL sequences")
    args = ap.parse_args()

    rows = []
    in_dir = args.raw_dir
    out_csv = args.out_csv
    files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".pdb")])
    if args.only:
        wanted = set(x.lower() for x in args.only)
        files = [f for f in files if os.path.splitext(f)[0].lower() in wanted]
    if not files:
        print(f"No PDBs found in {in_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit > 0:
        files = files[:args.limit]

    # Resume support: skip already processed IDs if requested
    processed = set()
    if args.resume and os.path.exists(out_csv):
        try:
            with open(out_csv, "r") as f:
                next(f, None)  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if parts:
                        processed.add(parts[0])
        except Exception:
            pass
        if processed:
            files = [f for f in files if os.path.splitext(f)[0] not in processed]

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(files, desc="Scanning PDBs")
    except Exception:
        iterator = files

    # Prepare writers (streaming writes so Ctrl-C keeps progress)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    csv_mode = "a" if (args.append and os.path.exists(out_csv)) else "w"
    need_header = True
    if csv_mode == "a":
        try:
            need_header = os.path.getsize(out_csv) == 0
        except Exception:
            need_header = True

    def _ensure_dir(p):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    fvh = None
    fvl = None
    if args.out_fasta_vh:
        _ensure_dir(args.out_fasta_vh)
        fa_mode_vh = "a" if (args.append and os.path.exists(args.out_fasta_vh)) else "w"
        fvh = open(args.out_fasta_vh, fa_mode_vh)
    if args.out_fasta_vl:
        _ensure_dir(args.out_fasta_vl)
        fa_mode_vl = "a" if (args.append and os.path.exists(args.out_fasta_vl)) else "w"
        fvl = open(args.out_fasta_vl, fa_mode_vl)

    rows_written = 0
    try:
        with open(out_csv, csv_mode, newline="") as fcsv:
            writer = csv.writer(fcsv)
            if need_header:
                writer.writerow(["pdb_id","vh_chain","vh_start","vh_end","vh_seq","vl_chain","vl_start","vl_end","vl_seq"])
                fcsv.flush()

            for fn in iterator:
                pdb_id = os.path.splitext(fn)[0]
                path = os.path.join(in_dir, fn)
                try:
                    vh, vl = process_pdb(
                        path,
                        scheme=args.scheme,
                        min_len=args.min_len,
                        debug=args.debug,
                        ncpu=args.ncpu,
                        species=args.species,
                        accept_heavy_only=args.accept_heavy_only,
                        accept_light_only=args.accept_light_only,
                        dump_raw=args.dump_anarci,
                        nterm_max=args.nterm_max,
                    )
                    if vh and vl:
                        vh_chain, vh_start, vh_end, _, vh_seq = vh
                        vl_chain, vl_start, vl_end, _, vl_seq = vl
                        row = (pdb_id, vh_chain, vh_start, vh_end, vh_seq, vl_chain, vl_start, vl_end, vl_seq)
                        rows.append(row)
                        writer.writerow(row)
                        fcsv.flush()
                        if fvh and vh_seq:
                            fvh.write(f">{pdb_id}|VH|{vh_chain}:{vh_start}-{vh_end}\n{vh_seq}\n")
                            fvh.flush()
                        if fvl and vl_seq:
                            fvl.write(f">{pdb_id}|VL|{vl_chain}:{vl_start}-{vl_end}\n{vl_seq}\n")
                            fvl.flush()
                        rows_written += 1
                        msg = f"OK {pdb_id}: VH {vh_chain}[{vh_start}:{vh_end}]  VL {vl_chain}[{vl_start}:{vl_end}]"
                        if args.show_seq:
                            msg += f"\n  VH seq: {vh_seq}\n  VL seq: {vl_seq}"
                        print(msg)
                    elif vh and not vl and args.accept_heavy_only:
                        vh_chain, vh_start, vh_end, _, vh_seq = vh
                        row = (pdb_id, vh_chain, vh_start, vh_end, vh_seq, "", "", "", "")
                        rows.append(row)
                        writer.writerow(row)
                        fcsv.flush()
                        if fvh and vh_seq:
                            fvh.write(f">{pdb_id}|VH|{vh_chain}:{vh_start}-{vh_end}\n{vh_seq}\n")
                            fvh.flush()
                        rows_written += 1
                        msg = f"OK {pdb_id}: VHH {vh_chain}[{vh_start}:{vh_end}] (heavy-only)"
                        if args.show_seq:
                            msg += f"\n  VH seq: {vh_seq}"
                        print(msg)
                    elif vl and not vh and args.accept_light_only:
                        vl_chain, vl_start, vl_end, _, vl_seq = vl
                        row = (pdb_id, "", "", "", "", vl_chain, vl_start, vl_end, vl_seq)
                        rows.append(row)
                        writer.writerow(row)
                        fcsv.flush()
                        if fvl and vl_seq:
                            fvl.write(f">{pdb_id}|VL|{vl_chain}:{vl_start}-{vl_end}\n{vl_seq}\n")
                            fvl.flush()
                        rows_written += 1
                        msg = f"OK {pdb_id}: VL {vl_chain}[{vl_start}:{vl_end}] (light-only)"
                        if args.show_seq:
                            msg += f"\n  VL seq: {vl_seq}"
                        print(msg)
                    else:
                        if args.debug:
                            print(f"SKIP {pdb_id}: could not find both VH and VL", file=sys.stderr)
                        else:
                            print(f"SKIP {pdb_id}: could not find both VH and VL")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FAIL {pdb_id}: {e}")
    finally:
        if fvh:
            try:
                fvh.close()
            except Exception:
                pass
        if fvl:
            try:
                fvl.close()
            except Exception:
                pass

    print("Wrote", out_csv, f"({rows_written} rows). Please spot-check a few.")

if __name__ == "__main__":
    main()
