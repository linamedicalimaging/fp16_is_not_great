import os
import sys

# -----------------------------
# Early GPU selection (before torch import)
# -----------------------------
DEFAULT_GPUS = "0,1,2,3,5,6"
# allow override from CLI: --gpus "0,1,2" or --gpus=0,1,2
_cli_gpus = None
for i, a in enumerate(sys.argv):
    if a == "--gpus" and i + 1 < len(sys.argv):
        _cli_gpus = sys.argv[i + 1]
        break
    if a.startswith("--gpus="):
        _cli_gpus = a.split("=", 1)[1]
        break
if _cli_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cli_gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS)

import csv
import math
import json
import time
import argparse
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer
#
import random
random.seed(2025)
# --- project imports ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(THIS_DIR), "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from Dataset.multi_dataset_test import multi_dataset
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM


def setup_logger():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def infer_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="/home/lina/GitHub/pytorch_model.bin",
                   help="Path to model checkpoint (.bin or directory).")
    p.add_argument("--lang_path", type=str,
                   default="/home/lina/GitHub/Llama-2-13b-hf",
                   help="Path to language model backbone (LLaMA directory).")
    p.add_argument("--tokenizer_path", type=str,
                   default="/home/lina/GitHub/Llama-2-13b-hf",
                   help="Path to tokenizer (must match LM family).")
    p.add_argument("--test_split", default="caption", type=str)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--num_workers", default=1, type=int)
    p.add_argument("--max_new_tokens", default=256, type=int)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--top_p", default=1.0, type=float)
    p.add_argument("--out_csv", default="/home/lina/RadFM_fine-tuning/Results_author_original_json/output_report_gen_INbreast_测试删除.csv", type=str)     # "/home/lina/RadFM_fine-tuning/Results_modify_json/output_report_gen_mimic_3.csv"
    p.add_argument("--no_add_image_tokens", action="store_true")
    p.add_argument("--gpus", default=None, type=str, help="Override CUDA_VISIBLE_DEVICES, e.g. '0,1,2'. (Must be set before torch import; handled early.)")
    return p.parse_args()


def build_tokenizer(path: str, add_image_tokens: bool = True):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    special_map = {}

    if tok.pad_token_id is None:
        special_map["pad_token"] = tok.eos_token if tok.eos_token else "<|pad|>"

    if add_image_tokens:
        extra = ["<image>", "</image>"]   # Keep only these two
        cur = set(tok.additional_special_tokens or [])
        add_list = [t for t in extra if t not in cur]
        if add_list:
            special_map.setdefault("additional_special_tokens", [])
            special_map["additional_special_tokens"].extend(add_list)
        if hasattr(tok, "unique_no_split_tokens"):
            tok.unique_no_split_tokens = list(set((tok.unique_no_split_tokens or []) + extra))

    if special_map:
        tok.add_special_tokens(special_map)
    return tok

def create_device_map(model, gpu_count: int):
    """Evenly shard LLaMA layers across logical GPUs 0..gpu_count-1.
    Anchor embeddings/norm/head and the custom embedding_layer on cuda:0.
    """
    from collections import OrderedDict
    dm = OrderedDict()
    dm["embedding_layer"] = 0
    dm["lang_model.model.embed_tokens"] = 0
    dm["lang_model.model.norm"] = 0
    dm["lang_model.lm_head"] = 0

    layers = model.lang_model.model.layers
    num_layers = len(layers)
    per = (num_layers + gpu_count - 1) // gpu_count
    for i in range(num_layers):
        dm[f"lang_model.model.layers.{i}"] = min(i // per, gpu_count - 1)
    return dm


def main():
    setup_logger()
    args = infer_args()

    # Report GPU visibility
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi‑GPU inference.")
    n_gpus = torch.cuda.device_count()
    logging.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f"Visible GPUs: {n_gpus}; using logical indices 0..{n_gpus-1}")

    logging.info("Loading tokenizer…")
    tok = build_tokenizer(args.tokenizer_path, add_image_tokens=(not args.no_add_image_tokens))

    logging.info("Initializing empty model graph…")
    with init_empty_weights():
        model = MultiLLaMAForCausalLM(lang_model_path=args.lang_path)

    logging.info("Creating device_map…")
    device_map = create_device_map(model, n_gpus)

    logging.info("Dispatching weights across GPUs…")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=args.ckpt,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"],
        offload_folder=None,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # resize token embeddings if we added special tokens
    try:
        model.lang_model.resize_token_embeddings(len(tok))
    except Exception:
        pass

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    logging.info("Preparing dataset & dataloader…")
    test_ds = multi_dataset(text_tokenizer=args.tokenizer_path, test_split=args.test_split)
    if len(test_ds) == 0:
        raise RuntimeError(f"No samples loaded for test_split={args.test_split}. "
                           f"Did you mean --test_split caption ?")

    dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
    )

    # Main device: Place the input on cuda:0 (the model has been sharded across multiple GPUs).
    main_device = torch.device("cuda:0")

    # Generation parameters (if you really need sampling/length parameters, you'll need to modify the model's generate signature to pass them through
    # currently MultiLLaMAForCausalLM.generate only accepts (lang_x, vision_x))
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )

    # Decide the autocast dtype in advance to avoid the "undefined" red line.
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype: torch.dtype = torch.bfloat16 if use_bf16 else torch.float16

    rows = []
    model.eval()
    torch.set_grad_enabled(False)

    logging.info("Starting multi-GPU inference…")
    for batch in dl:
        try:
            question = batch["question"]
            vision_x = batch["vision_x"]
            answer = batch.get("answer", "")
            belong_to = batch.get("belong_to", "")

            prompts = question if isinstance(question, list) else [question]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            input_ids = enc["input_ids"].to(main_device, non_blocking=True)
            attn_mask = enc.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(main_device, non_blocking=True)

            vision_x = vision_x.to(main_device, non_blocking=True)

            # Key point: Inside the loop, perform autocast + generate only after all data is ready.
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # RadFM's custom generate() only accepts two positional arguments: (lang_x, vision_x).
                out = model.generate(input_ids, vision_x)
                # If you've already added support for **gen_kwargs internally, you can modify it to:
                # out = model.generate(input_ids, vision_x, **gen_kwargs)

            # Parse the output
            if out.ndim == 2 and out.size(0) > 0:
                gen_tokens = out[0]
                # If the output includes the input, truncate the preceding prompt.
                if gen_tokens.size(0) >= input_ids.size(1):
                    gen_tokens = gen_tokens[input_ids.size(1):]
                pred = tok.decode(gen_tokens, skip_special_tokens=True)
            else:
                pred = tok.decode(out.reshape(-1), skip_special_tokens=True)

            q_str = prompts[0]
            gt_str = answer[0] if isinstance(answer, list) and len(answer) > 0 else (answer if isinstance(answer, str) else "")
            bt_str = belong_to[0] if isinstance(belong_to, list) and len(belong_to) > 0 else (belong_to if isinstance(belong_to, str) else "")
            rows.append([q_str, gt_str, pred, bt_str])

        except Exception:
            logging.error("A sample failed; printing traceback and continuing…")
            traceback.print_exc()
            continue


    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Ground Truth", "Pred", "belong_to"])
        w.writerows(rows)

    logging.info(f"Done. Saved → {out_csv}")


if __name__ == "__main__":
    main()
