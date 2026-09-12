"""Tokenize a text corpus into flat uint16 token streams for the text MDLM.

    python src/text_data.py --out /data/ryan.kim/registers_text_data

Writes <out>/{train,val,test}.npy (flat uint16 token ids) plus meta.json.
GPT-2 BPE has 50257 ids, which fits in uint16.
"""

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", default="/data/ryan.kim/registers_text_data")
    ap.add_argument("--num_proc", type=int, default=32)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(args.dataset, args.config)
    print({k: len(v) for k, v in ds.items()}, flush=True)

    eos = tok.eos_token_id
    assert tok.vocab_size <= 65536, "uint16 cannot hold this vocabulary"

    def encode(batch):
        ids = tok(batch["text"])["input_ids"]
        # one EOS between documents so the packed stream has boundaries
        return {"ids": [x + [eos] for x in ids], "len": [len(x) + 1 for x in ids]}

    enc = ds.map(encode, batched=True, remove_columns=["text"],
                 num_proc=args.num_proc, desc="tokenizing")

    split_map = {"train": "train", "validation": "val", "test": "test"}
    meta = {"dataset": args.dataset, "config": args.config,
            "tokenizer": args.tokenizer, "vocab_size": len(tok),
            "eos_token_id": eos, "counts": {}}
    for src, dst in split_map.items():
        if src not in enc:
            continue
        total = int(np.sum(enc[src]["len"]))
        path = os.path.join(args.out, f"{dst}.npy")
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16,
                                        shape=(total,))
        i = 0
        for row in enc[src]["ids"]:
            arr[i:i + len(row)] = row
            i += len(row)
        assert i == total, (i, total)
        arr.flush()
        del arr
        meta["counts"][dst] = total
        print(f"{dst}: {total:,} tokens -> {path}", flush=True)

    meta["wall_clock_s"] = time.time() - t0
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
