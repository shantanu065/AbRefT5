import torch
from transformers import T5EncoderModel, T5Tokenizer

class ProtT5Embedder(torch.nn.Module):
    def __init__(self, t5_name="Rostlab/prot_t5_xl_uniref50", freeze=True, device_map="auto"):
        super().__init__()
        # 1) New tokenizer arg to silence the legacy warning
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name, do_lower_case=False, legacy=False)

        # 2) Force safetensors + low mem, and optionally use device_map="auto" (needs accelerate installed)
        self.encoder = T5EncoderModel.from_pretrained(
            t5_name,
            use_safetensors=True,         # <<< force safetensors
            low_cpu_mem_usage=True,
            device_map=device_map if device_map else None,
            dtype=torch.float16 if torch.cuda.is_available() else None,
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, seqs: list[str]):
        # ProtT5 expects space-separated residues
        seqs = [" ".join(list(s.replace(" ", "").upper())) for s in seqs]
        device = next(self.encoder.parameters()).device
        batch = self.tokenizer(seqs, return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.encoder(**batch)
        return out.last_hidden_state  # [B, L, 1024]

