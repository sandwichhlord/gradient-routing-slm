"""Generate samples from the trained checkpoint -> out/samples.txt."""
import argparse
import os
import sys

import tiktoken
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, autocast_ctx, load_checkpoint, pick_device  # noqa: E402

PROMPTS = ["Once upon a time", "Q: 37 + 48 = A:", "def add(a, b):"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "out"))
    p.add_argument("--device", default="auto")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    model, ckpt = load_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), device)
    enc = tiktoken.get_encoding("gpt2")

    lines = [f"# samples from checkpoint @ iter {ckpt['iter']} "
             f"(temperature={args.temperature}, top_k={args.top_k}, max_new_tokens={args.max_new_tokens})\n"]
    for prompt in PROMPTS:
        x = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device).repeat(args.num_samples, 1)
        with autocast_ctx(device):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                               stop_token=enc.eot_token)
        for i in range(args.num_samples):
            toks = [t for t in y[i].tolist()]
            if enc.eot_token in toks[x.size(1):]:  # cut at first end-of-text after the prompt
                toks = toks[:x.size(1) + toks[x.size(1):].index(enc.eot_token)]
            text = enc.decode(toks)
            lines.append(f"=== prompt: {prompt!r} | sample {i + 1} ===\n{text}\n")
    out_path = os.path.join(args.out_dir, "samples.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[sample] wrote {out_path}")


if __name__ == "__main__":
    main()
