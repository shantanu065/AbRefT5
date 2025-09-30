#!/usr/bin/env python3
# scripts/data_fil.py
import os, csv, sys, argparse, re, io, shutil
from typing import List, Tuple, Optional, Dict
from contextlib import redirect_stderr
from Bio.PDB import PDBParser, Polypeptide
from Bio.SeqUtils import seq1
from anarci import run_anarci

RAW_DIR_DEFAULT = "data/raw_pdb"
OUT_CSV_DEFAULT = "data/chains.csv"
SCHEME_DEFAULT  = "chothia"  # or "imgt", "kabat", "aho"
SKIPPED_DIR_DEFAULT = "data/skipped_pdbs"
SKIPPED_LIST_DEFAULT = "data/skipped.txt"

CUSTOM_MAP_1L: Dict[str, str] = {
    "MSE":"M","FME":"M",
    "SEC":"C","PYL":"K",
    "SEP":"S","TPO":"T","PTR":"Y",
}

def chain_seq(chain) -> str:
    s = []
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
        s.append(aa)
    return "".join(s)

def chain_seq_and_index_map(chain) -> Tuple[str, List[Tuple[int, str, tuple]]]:
    seq_chars = []; idx_map = []
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
        idx_map.append((0, chain.id, res.get_id()))
    return "".join(seq_chars), idx_map

# ---------------- Header parsing ----------------
def parse_paired_hl(path: str) -> Tuple[set, set]:
    """Parse REMARK PAIRED_HL HCHAIN=, LCHAIN= from SAbDab-style headers."""
    hs, ls = set(), set()
    try:
        with open(path, 'r', errors='ignore') as fh:
            for line in fh:
                if line.startswith('ATOM'): break
                if not line.startswith('REMARK'): continue
                if 'PAIRED_HL' in line and ('HCHAIN' in line and 'LCHAIN' in line):
                    parts = re.split(r'\s+', line.strip())
                    for p in parts:
                        up = p.upper()
                        if up.startswith('HCHAIN') and ('=' in p or ':' in p):
                            v = p.split('=',1)[-1] if '=' in p else p.split(':',1)[-1]
                            if v: hs.add(v.strip())
                        if up.startswith('LCHAIN') and ('=' in p or ':' in p):
                            v = p.split('=',1)[-1] if '=' in p else p.split(':',1)[-1]
                            if v: ls.add(v.strip())
    except Exception:
        pass
    return hs, ls

def _find_chain_ids_near_token(text: str) -> List[str]:
    """Extract chain IDs near CHAIN / CHAIN: on a single header line."""
    ids = set()
    for m in re.finditer(r'CHAIN[:\s]+([A-Za-z0-9,\s]+)', text, flags=re.IGNORECASE):
        blob = m.group(1)
        for c in re.findall(r'[A-Za-z0-9]', blob):
            ids.add(c)
    for m in re.finditer(r'\bCHAIN\s+([A-Za-z0-9])\b', text, flags=re.IGNORECASE):
        ids.add(m.group(1))
    for m in re.finditer(r'\bCHAIN\s+([A-Za-z0-9])\|', text, flags=re.IGNORECASE):
        ids.add(m.group(1))
    for m in re.finditer(r'\(.*?\bCHAIN\s+([A-Za-z0-9])\b.*?\)', text, flags=re.IGNORECASE):
        ids.add(m.group(1))
    return list(ids)

def parse_chain_keywords(path: str) -> Tuple[set, set]:
    """
    Recognize header hints until first ATOM:
      - 'HEAVY CHAIN', 'LIGHT CHAIN'
      - 'VH', 'VL' tokens
      - '(HEAVY CHAIN|LIGHT CHAIN|VH|VL)' after 'Chain X ...'
      - **lambda/kappa inference**: if a header mentions LAMBDA or KAPPA and lists CHAIN ids,
        mark those chain ids as LIGHT.
    """
    hs, ls = set(), set()
    try:
        with open(path, 'r', errors='ignore') as fh:
            for raw in fh:
                if raw.startswith('ATOM'): break
                line = raw.strip()
                up = line.upper()

                # Parenthetical labels e.g., "Chain A ... (LIGHT CHAIN)" or "(VL)"
                m = re.search(r'\bCHAIN\s+([A-Za-z0-9])\b.*?\(([^)]+)\)', line, flags=re.IGNORECASE)
                if m:
                    cid = m.group(1)
                    par = m.group(2).upper()
                    if 'HEAVY CHAIN' in par or re.search(r'\bVH\b', par):
                        hs.add(cid)
                    if 'LIGHT CHAIN' in par or re.search(r'\bVL\b', par):
                        ls.add(cid)

                # Explicit HEAVY/LIGHT with chain list
                if 'HEAVY CHAIN' in up or 'LIGHT CHAIN' in up:
                    for c in _find_chain_ids_near_token(line):
                        if 'HEAVY CHAIN' in up: hs.add(c)
                        if 'LIGHT CHAIN' in up: ls.add(c)

                # VH/VL tokens with chain list
                if re.search(r'\bVH\b', up) or re.search(r'\bVL\b', up):
                    for c in _find_chain_ids_near_token(line):
                        if re.search(r'\bVH\b', up): hs.add(c)
                        if re.search(r'\bVL\b', up): ls.add(c)

                # NEW: lambda/kappa inference → LIGHT
                if ('LAMBDA' in up or 'KAPPA' in up) and 'CHAIN' in up:
                    for c in _find_chain_ids_near_token(line):
                        ls.add(c)
    except Exception:
        pass
    return hs, ls

def quick_find_HL(model):
    """Return (vh_tuple, vl_tuple) if H/L chain IDs are literally 'H'/'L' with plausible lengths."""
    H = None; L = None
    for chain in model:
        cid = str(chain.id)
        if cid.upper() in ("H","L"):
            full_seq = chain_seq(chain)
            if len(full_seq) >= 70:
                tup = (cid, 0, len(full_seq)-1, len(full_seq), full_seq, full_seq)
                if cid.upper()=="H": H = tup
                else: L = tup
    return H, L

# ---------- ANARCI wrappers & sanitizers ----------
def _anarci_key(label):
    pos, ins = label
    try:
        pos_i = int(pos)
    except Exception:
        pos_i = int(str(pos).strip() or 0)
    ins_code = 0 if (ins is None or ins == ' ') else (ord(ins[0]) if isinstance(ins, str) and ins else 0)
    return (pos_i, ins_code)

def _sanitize_numbering(numbering):
    cleaned = []
    last_key = None
    for item in numbering:
        label = None; letter = None
        try:
            label = item[0]; letter = item[2]
        except Exception:
            try:
                label = item[0]; letter = item[1]
            except Exception:
                continue
        if not label or not letter or letter == '-':
            continue
        k = _anarci_key(label)
        if (last_key is None) or (k >= last_key):
            cleaned.append((label, None, letter))
            last_key = k
    return cleaned

def _run_anarci_quiet(pairs, **kwargs):
    verbose = bool(kwargs.pop("_verbose_stderr", False))
    if verbose:
        return run_anarci(pairs, **kwargs)
    buf = io.StringIO()
    with redirect_stderr(buf):
        return run_anarci(pairs, **kwargs)

def _extract_anarci_domains(res_obj, debug: bool=False):
    domains = []
    try:
        results = res_obj[1] if (isinstance(res_obj, tuple) and len(res_obj) >= 2) else \
                  (res_obj[0] if (isinstance(res_obj, (list, tuple)) and len(res_obj) >= 1) else res_obj)
        if not results: return []
        seq_hits = results[0]
        if not seq_hits: return []
        if isinstance(seq_hits, (list, tuple)) and seq_hits and all(isinstance(x, (list, tuple)) for x in seq_hits):
            for hit in seq_hits:
                if len(hit) < 2: continue
                chain_type, details = hit[0], hit[1]
                doms = [details] if isinstance(details, dict) else [d for d in (details if isinstance(details, (list, tuple)) else []) if isinstance(d, dict)]
                for d in doms:
                    numbering = d.get("numbering")
                    if numbering is None: continue
                    domains.append({"chain_type": chain_type, "numbering": numbering, "start": d.get("start"), "end": d.get("end")})
        if not domains and isinstance(seq_hits, (list, tuple)) and seq_hits and all(isinstance(d, dict) for d in seq_hits):
            for d in seq_hits:
                numbering = d.get("numbering")
                if numbering is None: continue
                domains.append({"chain_type": d.get("chain_type"), "numbering": numbering, "start": d.get("start"), "end": d.get("end")})
        if not domains and isinstance(seq_hits, dict):
            numbering = seq_hits.get("numbering")
            if numbering is not None:
                domains.append({"chain_type": seq_hits.get("chain_type"), "numbering": numbering, "start": seq_hits.get("start"), "end": seq_hits.get("end")})

        for d in domains:
            numbering = d.get("numbering")
            if numbering is None:
                d["letters"] = None
                continue
            try:
                numbering_clean = _sanitize_numbering(numbering)
                d["numbering"] = numbering_clean
                d["letters"] = "".join(t[2] for t in numbering_clean)
            except Exception:
                try:
                    d["letters"] = "".join(item[2] if (isinstance(item, tuple) and len(item)>=3 and item[2] != "-") else "" for item in numbering)
                except Exception:
                    d["letters"] = None
    except Exception as e:
        if debug: print(f"ANARCI parse error: {e}", file=sys.stderr)
    return domains

# ---------- chain typing & patterns ----------
HEAVY_PREFIXES = ('QVQL','EVQL','QVQ','EVQ','QVEL','QLQL','QVTL','EVHL','QVHL')
LIGHT_PREFIXES = (
    'ELVMTQ','DIVMTQ','DIQMTQ','EIVLTQ','QSVLTQ','QLVLTQ','LVMTQ','AIQLTQ',
    'KLVMTQ','QIVLTQ','EVVLTQ','QSVLT','QPVLTQ','DIVLTQ','ELVLTQ','EIVMTQ'
)

def _guess_chain_type(domain_seq: str, chain_id: Optional[str] = None) -> Optional[str]:
    s = domain_seq.upper()
    if chain_id:
        if chain_id.upper() == 'H': return 'H'
        if chain_id.upper() == 'L': return 'L'
    if s.startswith(HEAVY_PREFIXES): return 'H'
    if s.startswith(LIGHT_PREFIXES): return 'L'
    if 'FGGGTK' in s or s.endswith('FGGGTKL') or re.search(r'F.GGTK[L]?', s): return 'L'
    if 'DYW' in s or 'WGQ' in s or 'WGAG' in s: return 'H'
    return None

def anarci_variable_span(seq: str, scheme="chothia", debug=False, ncpu=1, species=None, dump_raw=False):
    """
    Return (ctype, start, end, letters) or None.
    Explicitly set allowed_species=None unless species provided; suppress ANARCI stderr unless debug/dump_raw.
    """
    try:
        kwargs = {"scheme": scheme, "ncpu": max(1, int(ncpu)), "allowed_species": None}
        if species:
            kwargs["allowed_species"] = species
        kwargs["_verbose_stderr"] = bool(debug or dump_raw)
        res = _run_anarci_quiet([("chain", seq)], **kwargs)
        if dump_raw and debug:
            preview = str(res)
            print("ANARCI raw (trunc):\n" + (preview[:1200] + (" ...<trunc>" if len(preview)>1200 else "")), file=sys.stderr)
        doms = _extract_anarci_domains(res, debug=debug)
        if not doms:
            if debug: print("ANARCI: no domains detected", file=sys.stderr)
            return None
        best = max(doms, key=lambda d: len(d.get("letters") or ""))
        letters = best.get("letters") or ""
        ctype = best.get("chain_type") or _guess_chain_type(letters)
    except Exception as e:
        if debug: print(f"ANARCI failed: {e}", file=sys.stderr)
        return None
    if not letters: return None
    start = seq.find(letters)
    if start == -1: return None
    end = start + len(letters) - 1
    return (ctype, start, end, letters)

# ---------- Extra fallbacks ----------
def _looks_like_variable_len(n: int) -> bool:
    return 70 <= n <= 260

VH_START_RE = re.compile(r'(?:EVQL|QVQL|EVQ|QVQ|QVEL|QLQL|QVTL|EVHL|QVHL)')
VL_START_RE = re.compile(r'(?:EIVLTQ|ELVLTQ|ELVMTQ|EIVMTQ|DIVLTQ|DIVMTQ|DIQMTQ|QIVLTQ|KLVMTQ|QLVLTQ|QSVLTQ|QPVLTQ|EVVLTQ|LVMTQ|AIQLTQ|QSVLTQ)')
VL_END_RE   = re.compile(r'F.GGTK[L]?$')

def anchor_crop_variable(seq: str) -> Optional[Tuple[str,int,int,str]]:
    s = seq.upper()
    # VH crop
    mstart = VH_START_RE.search(s[:40])
    mend_candidates = [m for m in re.finditer(r'WG..', s)]
    if mstart and mend_candidates:
        for mend in reversed(mend_candidates):
            end = mend.end()
            length = end - mstart.start()
            if _looks_like_variable_len(length):
                v = s[mstart.start():end]
                return ('H', mstart.start(), end-1, v)
    # VL crop
    mstart = VL_START_RE.search(s[:50])
    mend = VL_END_RE.search(s)
    if mstart and mend:
        end = mend.end()
        length = end - mstart.start()
        if _looks_like_variable_len(length):
            v = s[mstart.start():end]
            return ('L', mstart.start(), end-1, v)
    # VL start present but no end motif → take ~110aa window
    if mstart and not mend:
        start = mstart.start()
        end = min(len(s), start + 110)
        if _looks_like_variable_len(end - start):
            v = s[start:end]
            return ('L', start, end-1, v)
    return None

def sliding_window_anarci(seq: str, scheme: str, debug: bool, ncpu: int, species, dump_raw: bool) -> Optional[Tuple[str,int,int,str]]:
    s = seq
    best: Optional[Tuple[str,int,int,str]] = None
    best_len = 0
    for win in (80, 90, 100, 110, 120, 130, 140, 150):
        if len(s) < win: continue
        for start in range(0, len(s) - win + 1, 10):
            sub = s[start:start+win]
            span = anarci_variable_span(sub, scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)
            if span:
                ctype, ss, ee, letters = span
                abs_s = start + ss
                abs_e = start + ee
                vlen = abs_e - abs_s + 1
                if vlen > best_len and _looks_like_variable_len(vlen):
                    best = (ctype, abs_s, abs_e, s[abs_s:abs_e+1])
                    best_len = vlen
    return best

# ---------- PDB processing ----------
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

    # Collect chain hints (explicit + lambda/kappa)
    paired_H, paired_L = parse_paired_hl(pdb_path)
    kw_H, kw_L = parse_chain_keywords(pdb_path)
    paired_H |= kw_H
    paired_L |= kw_L

    try:
        structure = parser.get_structure("x", pdb_path)
        model = next(structure.get_models())
    except Exception as e:
        if debug: print(f"PDBParser failed on {os.path.basename(pdb_path)}: {e}", file=sys.stderr)
        model = None

    vh_candidates = []
    vl_candidates = []

    def handle_chain(cid: str, full_seq: str):
        nonlocal vh_candidates, vl_candidates
        if len(full_seq) < min_len:
            return
        # If header provides pairing, restrict to those chains
        if (paired_H or paired_L) and (cid not in paired_H) and (cid not in paired_L):
            return

        # Try ANARCI on full seq
        span = anarci_variable_span(full_seq, scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)

        # Optional N-terminus cap
        if span is None and nterm_max > 0:
            if debug: print(f"ANARCI miss on {cid}; try N-terminus {nterm_max}", file=sys.stderr)
            span = anarci_variable_span(full_seq[:nterm_max], scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)

        # Sliding-window ANARCI
        if span is None:
            sw = sliding_window_anarci(full_seq, scheme=scheme, debug=debug, ncpu=ncpu, species=species, dump_raw=dump_raw)
            if sw is not None:
                span = sw
                if debug:
                    ctype_dbg, s_dbg, e_dbg, _ = span
                    print(f"Sliding-window ANARCI rescue on {os.path.basename(pdb_path)} chain {cid}: "
                          f"type={ctype_dbg}, span={s_dbg}-{e_dbg}", file=sys.stderr)

        # Anchor-based crop
        if span is None and _looks_like_variable_len(len(full_seq)):
            ac = anchor_crop_variable(full_seq)
            if ac is not None:
                span = ac
                if debug:
                    ctype_dbg, s_dbg, e_dbg, _ = span
                    print(f"Anchor-crop rescue on {os.path.basename(pdb_path)} chain {cid}: "
                          f"type={ctype_dbg}, span={s_dbg}-{e_dbg}", file=sys.stderr)

        # Sequence-based VL fallback (no header needed)
        if span is None and accept_light_only:
            up = full_seq.upper()
            mstart = VL_START_RE.search(up[:60])  # VL starts usually near N-terminus
            if mstart is not None:
                start = mstart.start()
                end = min(len(up), start + 110)
                if _looks_like_variable_len(end - start):
                    letters = up[start:end]
                    span = ('L', start, end - 1, letters)
                    if debug:
                        print(
                            f"Seq-VL fallback on {os.path.basename(pdb_path)} chain {cid}: "
                            f"start={start}, end={end-1}, len={end-start}",
                            file=sys.stderr
                        )

        # Heuristic whole-chain fallback (type guess)
        if span is None and _looks_like_variable_len(len(full_seq)):
            ctype_guess = _guess_chain_type(full_seq, chain_id=str(cid))
            if ctype_guess in ("H", "L"):
                s, e = 0, len(full_seq) - 1
                letters = full_seq
                span = (ctype_guess, s, e, letters)
                if debug:
                    print(f"Heuristic variable fallback on {os.path.basename(pdb_path)} chain {cid}: "
                          f"type={ctype_guess}, len={len(full_seq)}", file=sys.stderr)

        # If still nothing, but header says LIGHT and user allows VL-only, keep full chain
        if span is None:
            if (cid in paired_L) and accept_light_only:
                ctype_guess = "L"
                s, e = 0, len(full_seq) - 1
                letters = full_seq
                span = (ctype_guess, s, e, letters)
                if debug:
                    print(
                        f"Header-fallback (LIGHT CHAIN full sequence) on {os.path.basename(pdb_path)} chain {cid}: "
                        f"len={len(full_seq)}",
                        file=sys.stderr
                    )
            else:
                return

        # Record candidate
        ctype, s, e, letters = span
        vseq = full_seq[s:e+1]

        # Force ctype via header hints if present
        if cid in paired_H: ctype = 'H'
        if cid in paired_L: ctype = 'L'

        if ctype == "H":
            vh_candidates.append((cid, s, e, len(vseq), vseq, full_seq))
        elif ctype == "L":
            vl_candidates.append((cid, s, e, len(vseq), vseq, full_seq))
        else:
            g = _guess_chain_type(vseq, chain_id=str(cid))
            if g == "H":
                vh_candidates.append((cid, s, e, len(vseq), vseq, full_seq))
            elif g == "L":
                vl_candidates.append((cid, s, e, len(vseq), vseq, full_seq))

    # Parse the PDB
    if model is not None:
        # Quick path: literal chain ids 'H'/'L'
        vh_quick, vl_quick = quick_find_HL(model)
        if vh_quick and vl_quick:
            if (paired_H and vh_quick[0] not in paired_H) or (paired_L and vl_quick[0] not in paired_L):
                pass  # mismatch -> fall through to per-chain
            else:
                return vh_quick, vl_quick

        # Per-chain
        for chain in model:
            full_seq, _ = chain_seq_and_index_map(chain)
            handle_chain(chain.id, full_seq)
    else:
        # Manual text parse fallback
        seq_by_chain = {}
        with open(pdb_path, 'r', errors='ignore') as fh:
            last_key = None
            for line in fh:
                if not line.startswith('ATOM'): continue
                resname = line[17:20].strip()
                chain_id = (line[21].strip() or ' ')
                resseq_field = line[22:26].strip()
                icode = line[26].strip()
                key = (chain_id, resseq_field, icode)
                if key == last_key: continue
                last_key = key
                try:
                    aa = seq1(resname, custom_map=CUSTOM_MAP_1L)
                except Exception:
                    continue
                if not aa or len(aa) != 1 or aa.upper() == 'X': continue
                seq_by_chain.setdefault(chain_id, []).append(aa)
        for cid, aas in seq_by_chain.items():
            full_seq = "".join(aas)
            handle_chain(cid, full_seq)

    # Pick best candidates (longest variable span)
    vh_candidates.sort(key=lambda x: x[3], reverse=True)
    vl_candidates.sort(key=lambda x: x[3], reverse=True)

    vh = vh_candidates[0] if vh_candidates else None
    vl = vl_candidates[0] if vl_candidates else None

    if vh and vl: return vh, vl
    if vh and not vl and accept_heavy_only: return vh, None
    if vl and not vh and accept_light_only: return None, vl
    return vh if vh else None, vl if vl else None

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scheme", choices=["chothia","imgt","kabat","aho"], default=SCHEME_DEFAULT)
    ap.add_argument("--raw_dir", default=RAW_DIR_DEFAULT)
    ap.add_argument("--out_csv", default=OUT_CSV_DEFAULT)
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ncpu", type=int, default=(os.cpu_count() or 1))
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--accept_heavy_only", action="store_true")
    ap.add_argument("--accept_light_only", action="store_true")
    ap.add_argument("--dump_anarci", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--nterm_max", type=int, default=0)
    ap.add_argument("--show_seq", action="store_true")
    ap.add_argument("--out_fasta_vh", default=None)
    ap.add_argument("--out_fasta_vl", default=None)
    ap.add_argument("--skipped_dir", default=SKIPPED_DIR_DEFAULT)
    ap.add_argument("--skipped_list", default=SKIPPED_LIST_DEFAULT)
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.raw_dir) if f.lower().endswith(".pdb")])
    if args.only:
        need = set(x.lower() for x in args.only)
        files = [f for f in files if os.path.splitext(f)[0].lower() in need]
    if not files:
        print(f"No PDBs found in {args.raw_dir}", file=sys.stderr)
        sys.exit(1)
    if args.limit > 0:
        files = files[:args.limit]

    if args.resume and os.path.exists(args.out_csv):
        processed = set()
        try:
            with open(args.out_csv, "r") as f:
                next(f, None)
                for line in f:
                    parts = line.strip().split(",")
                    if parts: processed.add(parts[0])
        except Exception:
            processed = set()
        if processed:
            files = [f for f in files if os.path.splitext(f)[0] not in processed]

    # Prepare outputs
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if args.skipped_dir:
        os.makedirs(args.skipped_dir, exist_ok=True)

    csv_mode = "a" if (args.append and os.path.exists(args.out_csv)) else "w"
    need_header = True
    if csv_mode == "a":
        try: need_header = os.path.getsize(args.out_csv) == 0
        except Exception: need_header = True

    def _ensure_dir(p): 
        d = os.path.dirname(p)
        if d: os.makedirs(d, exist_ok=True)

    fvh = fvl = None
    if args.out_fasta_vh:
        _ensure_dir(args.out_fasta_vh)
        fvh = open(args.out_fasta_vh, "a" if (args.append and os.path.exists(args.out_fasta_vh)) else "w")
    if args.out_fasta_vl:
        _ensure_dir(args.out_fasta_vl)
        fvl = open(args.out_fasta_vl, "a" if (args.append and os.path.exists(args.out_fasta_vl)) else "w")

    rows_written = 0
    try:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(files, desc="Scanning PDBs")
        except Exception:
            iterator = files

        with open(args.out_csv, csv_mode, newline="") as fcsv:
            writer = csv.writer(fcsv)
            if need_header:
                writer.writerow([
                    "pdb_id",
                    "vh_chain","vh_start","vh_end","vh_seq","vh_full_seq",
                    "vl_chain","vl_start","vl_end","vl_seq","vl_full_seq"
                ])
                fcsv.flush()

            for fn in iterator:
                pdb_id = os.path.splitext(fn)[0]
                path = os.path.join(args.raw_dir, fn)
                try:
                    res = process_pdb(
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
                    rows_local = []
                    msg = None
                    if isinstance(res, tuple) and len(res) == 2:
                        vh, vl = res
                        if vh and vl:
                            vh_chain, vh_start, vh_end, _, vh_seq, vh_full = vh
                            vl_chain, vl_start, vl_end, _, vl_seq, vl_full = vl
                            rows_local.append((pdb_id, vh_chain, vh_start, vh_end, vh_seq, vh_full,
                                               vl_chain, vl_start, vl_end, vl_seq, vl_full))
                            msg = f"OK {pdb_id}: VH {vh_chain}[{vh_start}:{vh_end}]  VL {vl_chain}[{vl_start}:{vl_end}]"
                        elif vh and not vl and args.accept_heavy_only:
                            vh_chain, vh_start, vh_end, _, vh_seq, vh_full = vh
                            rows_local.append((pdb_id, vh_chain, vh_start, vh_end, vh_seq, vh_full,
                                               "", "", "", "", ""))
                            msg = f"OK {pdb_id}: VHH {vh_chain}[{vh_start}:{vh_end}] (heavy-only)"
                        elif vl and not vh and args.accept_light_only:
                            vl_chain, vl_start, vl_end, _, vl_seq, vl_full = vl
                            rows_local.append((pdb_id, "", "", "", "", "",
                                               vl_chain, vl_start, vl_end, vl_seq, vl_full))
                            msg = f"OK {pdb_id}: VL {vl_chain}[{vl_start}:{vl_end}] (light-only)"

                    for row in rows_local:
                        writer.writerow(row)
                        fcsv.flush()
                        rows_written += 1
                        if fvh and row[4]:
                            fvh.write(f">{row[0]}|VH|{row[1]}:{row[2]}-{row[3]}\n{row[4]}\n"); fvh.flush()
                        if fvl and row[9]:
                            fvl.write(f">{row[0]}|VL|{row[6]}:{row[7]}-{row[8]}\n{row[9]}\n"); fvl.flush()

                    if msg:
                        if args.show_seq and rows_local:
                            r = rows_local[0]
                            vh_seq = r[4]; vl_seq = r[9]
                            if vh_seq: msg += f"\n  VH seq: {vh_seq}"
                            if vl_seq: msg += f"\n  VL seq: {vl_seq}"
                        print(msg)
                    else:
                        out = f"SKIP {pdb_id}: could not find both VH and VL"
                        # Save skipped metadata/files
                        try:
                            if args.skipped_list:
                                with open(args.skipped_list, "a") as fskip:
                                    fskip.write(pdb_id + "\n")
                            if args.skipped_dir:
                                shutil.copy(path, os.path.join(args.skipped_dir, fn))
                        except Exception:
                            pass
                        if args.debug:
                            print(out, file=sys.stderr)
                        else:
                            print(out)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"FAIL {pdb_id}: {repr(e)}", file=sys.stderr)

    finally:
        for fh in (fvh, fvl):
            try:
                if fh: fh.close()
            except Exception:
                pass

    print("Wrote", args.out_csv, f"({rows_written} rows). Please spot-check a few.")

if __name__ == "__main__":
    main()
