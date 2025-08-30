# Fine-tune RadFM: inject LoRA into language side (LLaMA) q/k/v/o only and train LoRA params on the author's checkpoint
import os
import re
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

# ===== Edit to your local paths =====
BASE_DIR_LLAMA = "/home/lina/GitHub/llama-13b-hf"           # directory containing config.json
AUTHOR_CKPT    = "/home/lina/GitHub/pytorch_model.bin"      # author-provided .bin
TOKENIZER_DIR  = "/home/lina/GitHub/llama-13b-hf"           # tokenizer path
OUTPUT_DIR     = "/home/lina/RadFM_fine-tuning/fine-tuning_result_llama1_base"
BATCH_SIZE     = 2
NUM_EPOCHS     = 3
LR             = 1e-4
WEIGHT_DECAY   = 0.0
LORA_RANK      = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
WARMUP_RATIO   = 0.03  # linear warmup ratio (of total steps)

# ====== Dataset / Collator ======
from Dataset.multi_dataset import multi_dataset as TrainDataset

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

# ===================== LoRA module =====================
class LoRALayer(nn.Module):
    """Wrap an nn.Linear with LoRA (low-rank A@B delta). Train A/B only; freeze the original Linear."""
    def __init__(self, original_layer: nn.Linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        assert isinstance(original_layer, nn.Linear), "LoRALayer only supports nn.Linear."
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original Linear
        for p in self.original_layer.parameters():
            p.requires_grad = False

        in_features  = original_layer.in_features
        out_features = original_layer.out_features
        device = original_layer.weight.device
        dtype  = original_layer.weight.dtype

        # LoRA A/B
        self.lora_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # Init: B=0 so the initial output is unchanged
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))

    @torch.no_grad()
    def merge(self):
        delta = (self.scaling * self.lora_b.weight) @ self.lora_a.weight
        self.original_layer.weight += delta

    @torch.no_grad()
    def unmerge(self):
        delta = (self.scaling * self.lora_b.weight) @ self.lora_a.weight
        self.original_layer.weight -= delta

# Only match language-side q/k/v/o of LLaMA (e.g., lang_model.model.layers.3.self_attn.q_proj)
_QKVO_RE = re.compile(
    r"^lang_model\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
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

def apply_lora_to_model(model: nn.Module, rank=8, alpha=16, dropout=0.05, verbose=True):
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _QKVO_RE.match(name):
            to_replace.append((name, module))
    for name, module in to_replace:
        parent, attr = _get_parent_and_attr(model, name)
        lora = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
        with torch.no_grad():
            lora.original_layer.weight.copy_(module.weight.data)
        setattr(parent, attr, lora)
        if verbose:
            print(f"[LoRA] injected -> {name} ({module.in_features}->{module.out_features})")
    print(f"[LoRA] total injected: {len(to_replace)}")
    return model

def freeze_non_lora(model: nn.Module):
    for n, p in model.named_parameters():
        if (".lora_a.weight" in n) or (".lora_b.weight" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

def extract_lora_state_dict(model: nn.Module):
    sd = model.state_dict()
    return {k: v for k, v in sd.items()
            if (".lora_a.weight" in k or ".lora_b.weight" in k)}

# ===================== Training script =====================
def assert_hidden_alignment(model: nn.Module, expect: int):
    if hasattr(model, "embedding_layer"):
        m = model.embedding_layer
        assert m.fc.out_features == expect, f"embedding_layer.fc.out={m.fc.out_features}, expect {expect}"
        if hasattr(m, "bert_projection_fc"):
            assert m.bert_projection_fc.out_features == expect, \
                f"bert_projection_fc.out={m.bert_projection_fc.out_features}, expect {expect}"

def main():
    accelerator = Accelerator()
    device = accelerator.device

    from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

    # 1) Build model skeleton
    model = MultiLLaMAForCausalLM(lang_model_path=BASE_DIR_LLAMA)

    # 2) Load author checkpoint
    state = torch.load(AUTHOR_CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    # 3) (Optional) hidden size alignment check
    try:
        expect_hidden = getattr(getattr(model, "lang_model", None), "config", None)
        expect_hidden = getattr(expect_hidden, "hidden_size", 5120) or 5120
        assert_hidden_alignment(model, expect_hidden)
    except Exception as e:
        print("[warn] hidden alignment check skipped:", e)

    # 4) Disable gradient checkpointing to avoid device mismatch & no-grad warnings
    try:
        model.lang_model.gradient_checkpointing_disable()
    except Exception:
        pass
    if hasattr(model.lang_model, "config"):
        model.lang_model.config.gradient_checkpointing = False
        model.lang_model.config.use_cache = False

    # 5) Move the whole model to device (critical)
    model = model.to(device)

    # 6) Inject LoRA and train LoRA only
    apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    freeze_non_lora(model)

    # 7) Data
    train_dataset = TrainDataset(text_tokenizer=TOKENIZER_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=DataCollator(),
    )

    # 8) Optimizer & Scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

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

    # 10) Train
    model.train()
    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Safely pass only the fields needed by forward
                kwargs = {}
                if "attention_mask" in batch:  kwargs["attention_mask"]  = batch["attention_mask"]
                if "loss_reweight"  in batch:  kwargs["loss_reweight"]   = batch["loss_reweight"]
                if "key_words_query" in batch: kwargs["key_words_query"] = batch["key_words_query"]

                outputs = model(
                    vision_x=batch.get("vision_x", None),
                    lang_x=batch.get("lang_x", None),
                    labels=batch.get("labels", None),
                    **kwargs
                )

                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise RuntimeError("Model did not return `loss`. Please check that labels and required inputs are provided correctly.")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                if step % 100 == 0:
                    accelerator.print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    # 11) Save LoRA only
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        raw_model = accelerator.unwrap_model(model)
        lora_sd = extract_lora_state_dict(raw_model)
        save_path = os.path.join(OUTPUT_DIR, "lora_lang_qkvo_weights.pth")
        torch.save(lora_sd, save_path)
        print(f"[save] LoRA weights -> {save_path}")

if __name__ == "__main__":
    main()
