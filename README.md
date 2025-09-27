# AbRefT5

**AbRefT5** is a fine-tuned **ProtT5** model for **antibody structure refinement**. It consumes VH/VL sequences (plus an optional initial structure) and predicts residue-frame corrections and torsion updates to improve backbone/side-chain quality.

> **One-liner:** AbRefT5 is a fine-tuned ProtT5 model designed for antibody structure refinement, leveraging protein language model embeddings to improve structural accuracy and conformational quality.

---

## 🔧 Features
- VH/VL-aware refinement with ProtT5 embeddings
- Backbone frame (ΔR, Δt) and torsion (φ/ψ/ω/χ) heads
- Lightweight stereochemistry and clash penalties
- Optional antigen-aware weighting for interface residues
- Fast inference; integrates with RFdiffusion/AF2 pipelines

## 📦 Installation
```bash
conda env create -f env.yml
conda activate abrefT5
# or: pip install -r requirements.txt
```
🚀 Quickstart
```
python -m abrefT5.inference \
  --vh_seq QVQLVESGGGLVQAGGSLRLSCAASG... \
  --vl_seq ELVMTQSPASLSVSVGETVTITCRAS... \
  --init_pdb path/to/initial.pdb \
  --out_pdb refined.pdb
```
📊 Metrics to track

  lDDT-Cα, backbone RMSD (FR vs CDRs)

  χ1/χ2 accuracy, clashscore, MolProbity

  (Optional) interface RMSD/fnat if antigen present

🔬 Citation

If you use AbRefT5, please cite this repository (preprint in preparation).
📜 License

MIT (see LICENSE).
