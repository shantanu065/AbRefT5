import argparse
import torch
from .models.protT5 import ProtT5Embedder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vh_seq", required=True)
    p.add_argument("--vl_seq", required=True)
    args = p.parse_args()

    model = ProtT5Embedder()
    model.eval()
    with torch.no_grad():
        emb = model([args.vh_seq, args.vl_seq])
    print("Embedding shape:", emb.shape)

if __name__ == "__main__":
    main()
