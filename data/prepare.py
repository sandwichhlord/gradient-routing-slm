"""Build the 3-domain corpus (story / math / code) with the GPT-2 tokenizer.

Outputs (in --out_dir, default data/out/):
  train.bin         uint16 tokens, documents from all domains shuffled together
  train_domain.bin  uint8 domain id per train token (0=story, 1=math, 2=code), for later routing work
  val_story.bin, val_math.bin, val_code.bin   held-out uint16 tokens per domain
  meta.json         token counts, mode (real / synthetic / synthetic_fallback), args

Every document is followed by the GPT-2 <|endoftext|> token.

Usage:
  python data/prepare.py                       # real data (TinyStories + synthetic math + codeparrot)
  python data/prepare.py --synthetic           # tiny offline fake data for all 3 domains
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import tiktoken

DOMAINS = ["story", "math", "code"]

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, fits in uint16


def tokenize_docs(texts):
    """Tokenize a list of strings; return list of np.uint16 arrays, each ending in EOT."""
    out = []
    for ids in enc.encode_ordinary_batch(texts, num_threads=8):
        ids.append(EOT)
        out.append(np.asarray(ids, dtype=np.uint16))
    return out


def collect(text_iter, budget, max_doc_tokens=None, batch=256, name=""):
    """Pull texts from an iterator, tokenize in batches, stop once `budget` tokens are collected."""
    docs, total, buf, t0 = [], 0, [], time.time()
    last_report = 0

    def flush():
        nonlocal total
        for d in tokenize_docs(buf):
            if max_doc_tokens and len(d) > max_doc_tokens:
                d = np.concatenate([d[: max_doc_tokens - 1], np.array([EOT], dtype=np.uint16)])
            docs.append(d)
            total += len(d)
        buf.clear()

    for text in text_iter:
        if not text or not text.strip():
            continue
        buf.append(text)
        if len(buf) >= batch:
            flush()
            if total - last_report >= 2_000_000:
                last_report = total
                print(f"  [{name}] {total/1e6:.1f}M / {budget/1e6:.1f}M tokens ({time.time()-t0:.0f}s)", flush=True)
            if total >= budget:
                break
    if buf and total < budget:
        flush()
    if total < budget:
        raise RuntimeError(f"[{name}] source exhausted at {total} tokens < budget {budget}")
    return docs


def split_val(docs, val_tokens):
    """Take documents from the end of the list until val_tokens is reached."""
    val, n = [], 0
    while docs and n < val_tokens:
        d = docs.pop()
        val.append(d)
        n += len(d)
    return docs, val


# ----------------------------------------------------------------------------- math (owned generator)

def math_line(rng):
    def num():
        digits = rng.randint(1, 3)
        return rng.randint(0, 10 ** digits - 1)
    a, b = num(), num()
    op = rng.choice(["+", "-", "x"])
    ans = a + b if op == "+" else a - b if op == "-" else a * b
    return f"Q: {a} {op} {b} = A: {ans}"


def math_docs_iter(rng, lines_per_doc=20):
    while True:
        yield "\n".join(math_line(rng) for _ in range(lines_per_doc))


# ----------------------------------------------------------------------------- HF sources

def hf_iter(path, field, split="train"):
    from datasets import load_dataset
    ds = load_dataset(path, split=split, streaming=True)
    for ex in ds:
        yield ex[field]


# ----------------------------------------------------------------------------- synthetic fallback

NAMES = ["Lily", "Tom", "Max", "Sara", "Ben", "Mia", "Leo", "Anna", "Sam", "Zoe"]
ANIMALS = ["cat", "dog", "bird", "frog", "bunny", "fox", "bear", "duck"]
THINGS = ["ball", "kite", "box", "hat", "cake", "toy", "book", "flower"]
PLACES = ["park", "garden", "forest", "house", "beach", "school", "farm"]
FEELINGS = ["happy", "sad", "scared", "excited", "tired", "proud"]


def fake_story(rng):
    n, a, t, p, f = (rng.choice(x) for x in (NAMES, ANIMALS, THINGS, PLACES, FEELINGS))
    n2 = rng.choice([x for x in NAMES if x != n])
    s = [
        f"Once upon a time, there was a little {a} named {n}.",
        f"{n} liked to play in the {p} with a red {t}.",
        f"One day, {n} lost the {t} and felt very {f}.",
        f"{n2} came and said, \"Do not worry, I will help you.\"",
        f"They looked under the trees and found the {t}.",
        f"{n} smiled and said thank you. They played together all day.",
        "The end.",
    ]
    return " ".join(s[: rng.randint(4, len(s))])


FUNCS = ["add", "sub", "mul", "scale", "clamp", "total", "average", "count_items"]
VARS = ["a", "b", "x", "y", "n", "value", "items", "data"]


def fake_code(rng):
    fn, v1, v2 = rng.choice(FUNCS), rng.choice(VARS), rng.choice(VARS)
    if v1 == v2:
        v2 = v2 + "2"
    k = rng.randint(1, 9)
    body = rng.choice([
        f"    return {v1} + {v2}\n",
        f"    result = {v1} * {k}\n    return result - {v2}\n",
        f"    if {v1} > {v2}:\n        return {v1}\n    return {v2}\n",
        f"    out = []\n    for i in range({v1}):\n        out.append(i * {k})\n    return out\n",
    ])
    return f"import math\n\n\ndef {fn}({v1}, {v2}):\n    \"\"\"Compute {fn}.\"\"\"\n{body}\n\nprint({fn}({k}, {k + 1}))\n"


def fake_iter(fn, rng):
    while True:
        yield fn(rng)


# ----------------------------------------------------------------------------- main

def write_bin(path, docs):
    arr = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint16)
    arr.astype(np.uint16).tofile(path)
    return len(arr)


def build(args, synthetic):
    rng = random.Random(args.seed)
    if synthetic:
        n = args.synthetic_tokens
        budgets = {d: n for d in DOMAINS}
        val_tokens = max(n // 10, 2048)
    else:
        budgets = {"story": args.story_tokens, "math": args.math_tokens, "code": args.code_tokens}
        val_tokens = args.val_tokens

    sources = {
        "math": lambda: math_docs_iter(rng),
        "story": (lambda: fake_iter(fake_story, rng)) if synthetic
        else (lambda: hf_iter("roneneldan/TinyStories", "text")),
        "code": (lambda: fake_iter(fake_code, rng)) if synthetic
        else (lambda: hf_iter("codeparrot/codeparrot-clean-valid", "content")),
    }
    train_docs, val_docs = {}, {}
    for d in DOMAINS:
        t0 = time.time()
        print(f"[prepare] collecting {d}: {budgets[d] + val_tokens:,} tokens", flush=True)
        docs = collect(sources[d](), budgets[d] + val_tokens,
                       max_doc_tokens=args.max_doc_tokens if d == "code" else None, name=d)
        train_docs[d], val_docs[d] = split_val(docs, val_tokens)
        print(f"[prepare] {d}: {len(docs)} train docs, {len(val_docs[d])} val docs ({time.time()-t0:.0f}s)", flush=True)
    return train_docs, val_docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    p.add_argument("--story_tokens", type=int, default=20_000_000)
    p.add_argument("--math_tokens", type=int, default=10_000_000)
    p.add_argument("--code_tokens", type=int, default=10_000_000)
    p.add_argument("--val_tokens", type=int, default=500_000)
    p.add_argument("--max_doc_tokens", type=int, default=4096, help="truncate very long code files")
    p.add_argument("--synthetic", action="store_true", help="tiny offline fake data for all 3 domains")
    p.add_argument("--synthetic_tokens", type=int, default=200_000, help="train tokens per domain in synthetic mode")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--force", action="store_true", help="rebuild even if matching data already exists")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    signature = {k: v for k, v in vars(args).items() if k not in ("out_dir", "force")}
    meta_path = os.path.join(args.out_dir, "meta.json")
    needed = ["train.bin", "train_domain.bin"] + [f"val_{d}.bin" for d in DOMAINS]
    if not args.force and os.path.exists(meta_path) and all(os.path.exists(os.path.join(args.out_dir, f)) for f in needed):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("signature") == signature and meta.get("mode") != "synthetic_fallback":
            print(f"[prepare] data in {args.out_dir} already matches these args ({meta['mode']}); skipping. Use --force to rebuild.")
            return

    mode = "synthetic" if args.synthetic else "real"
    try:
        train_docs, val_docs = build(args, synthetic=args.synthetic)
    except Exception as e:  # network / HF problems -> keep the pipeline alive offline
        if args.synthetic:
            raise
        print("\n" + "!" * 78, file=sys.stderr)
        print(f"[prepare] WARNING: real data download failed ({type(e).__name__}: {e})", file=sys.stderr)
        print("[prepare] FALLING BACK TO TINY SYNTHETIC DATA. Results will NOT be meaningful.", file=sys.stderr)
        print("!" * 78 + "\n", file=sys.stderr, flush=True)
        mode = "synthetic_fallback"
        train_docs, val_docs = build(args, synthetic=True)

    # shuffle documents across domains into one training stream, tracking each token's domain
    rng = np.random.default_rng(args.seed)
    tagged = [(i, doc) for i, d in enumerate(DOMAINS) for doc in train_docs[d]]
    order = rng.permutation(len(tagged))
    docs = [tagged[j][1] for j in order]
    dom = [np.full(len(tagged[j][1]), tagged[j][0], dtype=np.uint8) for j in order]
    n_train = write_bin(os.path.join(args.out_dir, "train.bin"), docs)
    np.concatenate(dom).tofile(os.path.join(args.out_dir, "train_domain.bin"))

    meta = {"mode": mode, "signature": signature, "vocab_size": enc.n_vocab, "eot": EOT,
            "domains": DOMAINS, "train_tokens": n_train,
            "train_tokens_per_domain": {d: int(sum(len(x) for x in train_docs[d])) for d in DOMAINS}}
    for d in DOMAINS:
        meta[f"val_{d}_tokens"] = write_bin(os.path.join(args.out_dir, f"val_{d}.bin"), val_docs[d])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] done ({mode}): {json.dumps({k: v for k, v in meta.items() if k != 'signature'})}")


if __name__ == "__main__":
    main()
