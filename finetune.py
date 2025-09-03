#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
GPU_REQUIRED = [0,1,2,3,4,5]
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_REQUIRED)[1:-1]
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from tensorboardX import SummaryWriter


# In[2]:

writer = SummaryWriter('runs')
BASE_DIR_LLAMA = "/home/lina/GitHub/llama-13b-hf"           # directory containing config.json
AUTHOR_CKPT    = "/home/lina/GitHub/pytorch_model.bin"      # author-provided .bin
TOKENIZER_DIR  = "/home/lina/GitHub/llama-13b-hf"           # tokenizer path
OUTPUT_DIR     = "/home/lina/RadFM_fine-tuning/fine-tuning_result_llama1_base"
BATCH_SIZE     = 2
NUM_EPOCHS     = 20
LR             = 1e-4
WEIGHT_DECAY   = 0.0
LORA_RANK      = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
WARMUP_RATIO   = 0.03  # linear warmup ratio (of total steps)

# ====== Dataset / Collator ======
from Dataset.multi_dataset import multi_dataset as TrainDataset


# In[3]:


class DataCollator(object):
    """A simplified implementation equivalent to the author's DataCollator in train.py."""
    def __call__(self, instances):
        vision_xs, lang_xs, attention_masks, labels, loss_reweight, key_words_query = tuple(
            [inst[k] for inst in instances]
            for k in ("vision_x", "lang_x", "attention_mask", "labels", "loss_reweight", "key_words_query")
        )

        import torch.nn.functional as F

        lang_xs        = torch.cat([x.unsqueeze(0) for x in lang_xs], dim=0)
        attention_mask = torch.cat([x.unsqueeze(0) for x in attention_masks], dim=0)
        labels         = torch.cat([x.unsqueeze(0) for x in labels], dim=0)
        loss_reweight  = torch.cat([x.unsqueeze(0) for x in loss_reweight], dim=0)

        target_H, target_W, target_D = 512, 512, 4
        MAX_D = 0
        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for v in vision_xs:
            try:
                D = v.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except Exception:
                continue

        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [F.interpolate(v, size=(target_H, target_W, target_D)) for v in vision_xs]
        vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs, batch_first=True, padding_value=0)

        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_mask,
            labels=labels,
            loss_reweight=loss_reweight,
            key_words_query=key_words_query,
        )


def _get_parent_and_attr(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def extract_lora_state_dict(model: nn.Module):
    sd = model.state_dict()
    return {k: v for k, v in sd.items()
            if (".lora_a.weight" in k or ".lora_b.weight" in k)}


# In[6]:


def assert_hidden_alignment(model: nn.Module, expect: int):
    if hasattr(model, "embedding_layer"):
        m = model.embedding_layer
        assert m.fc.out_features == expect, f"embedding_layer.fc.out={m.fc.out_features}, expect {expect}"
        if hasattr(m, "bert_projection_fc"):
            assert m.bert_projection_fc.out_features == expect, \
                f"bert_projection_fc.out={m.bert_projection_fc.out_features}, expect {expect}"

def create_device_map(model, gpu_count: int):
    """Evenly shard LLaMA layers across logical GPUs 0..gpu_count-1.
    Anchor embeddings/norm/head and the custom embedding_layer on cuda:0.
    """
    from collections import OrderedDict
    dm = OrderedDict()
    dm["embedding_layer"] = GPU_REQUIRED[0]
    dm["lang_model.model.embed_tokens"] = GPU_REQUIRED[0]
    dm["lang_model.model.norm"] = GPU_REQUIRED[0]
    dm["lang_model.lm_head"] = GPU_REQUIRED[0]

    layers = model.lang_model.model.layers
    num_layers = len(layers)
    per = (num_layers + gpu_count - 1) // gpu_count
    for i in range(num_layers):
        dm[f"lang_model.model.layers.{i}"] = min(i // per, gpu_count - 1)
    return dm


# In[7]:


with init_empty_weights():
    model = MultiLLaMAForCausalLM(BASE_DIR_LLAMA)
device_map = create_device_map(model, len(GPU_REQUIRED))
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=AUTHOR_CKPT,
    device_map=device_map,
    no_split_module_classes=["LlamaDecoderLayer"],
    offload_folder=None,)
model.add_freeze_non_lora(LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

total_params = sum([p.numel() for p in model.parameters()])
trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(
  f"""
  {total_params} total params,
  {trainable_params}" trainable params,
  {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
  """
)

train_dataset = TrainDataset(text_tokenizer=TOKENIZER_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=DataCollator(),
)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

accelerator = Accelerator()
grad_accum = getattr(accelerator, "gradient_accumulation_steps", 1)
total_steps = NUM_EPOCHS * math.ceil(len(train_loader) / grad_accum)
warmup_steps = max(10, int(WARMUP_RATIO * total_steps))

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# 9) Accelerator prepare (handles device placement and DDP)
model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

model.train()
step = 0
for epoch in range(NUM_EPOCHS):
    for batch in tqdm(train_loader):
        # Safely pass only the fields needed by forward
        kwargs = {}
        if "attention_mask" in batch:  kwargs["attention_mask"]  = batch["attention_mask"]
        if "loss_reweight"  in batch:  kwargs["loss_reweight"]   = batch["loss_reweight"]
        if "key_words_query" in batch: kwargs["key_words_query"] = batch["key_words_query"]

        loss = model(
            vision_x=batch.get("vision_x", None).to(GPU_REQUIRED[0]),
            lang_x=batch.get("lang_x", None).to(GPU_REQUIRED[0]),
            labels=batch.get("labels", None).to(GPU_REQUIRED[0]),
            **kwargs
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        step += 1
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
        if step % 30 == 0:
            writer.add_scalar('Loss/train', loss, step)

raw_model = accelerator.unwrap_model(model)
lora_sd = extract_lora_state_dict(raw_model)
torch.save(lora_sd, "lora_lang_qkvo_weights.pth")
