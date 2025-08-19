# import os       # Lina added
# import sys       # Lina added
#
# DEFAULT_GPUS = "0,1,2,3,5,6"  # Lina added
# _cli_gpus = None              # Lina added
# for i, a in enumerate(sys.argv):  # Lina added
#     if a == "--gpus" and i + 1 < len(sys.argv):
#         _cli_gpus = sys.argv[i + 1]
#         break
#     if a.startswith("--gpus="):
#         _cli_gpus = a.split("=", 1)[1]
#         break
# if _cli_gpus:
#     os.environ["CUDA_VISIBLE_DEVICES"] = _cli_gpus
# else:
#     os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS)
# # ------------------------------------------------------------
#
# import tqdm.auto as tqdm
# import torch.nn.functional as F
# from typing import Optional, Dict, Sequence
# from typing import List, Optional, Tuple, Union
# import transformers
# from My_Trainer.trainer import Trainer
# from dataclasses import dataclass, field
# from Dataset.multi_dataset import multi_dataset
# from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
# from datasampler import My_DistributedBatchSampler
# # original code:
# # from datasets import load_metric
# # from Dataset.multi_dataset_test_for_close import multi_dataset_close  # Lina removed (unused)
# import numpy as np
# import torch
#
# # === 新增：PEFT / LoRA ===
# from peft import LoraConfig, get_peft_model, TaskType  # Lina added
#
#
# def compute_metrics(eval_preds):
#     ACCs = eval_preds.predictions
#     return {"accuracy": np.mean(ACCs, axis=-1)}
#
#
# @dataclass
# class ModelArguments:
#     lang_encoder_path: Optional[str] = field(
#         default="/home/lina/GitHub/Llama-2-13b-hf")
#     tokenizer_path: str = field(default='/home/lina/GitHub/Llama-2-13b-hf',
#                                 metadata={"help": "Path to the tokenizer data."})
#
#
# @dataclass
# class DataArguments:
#     Mode: Optional[str] = field(default="Train")
#     # Lina added: 只跑“诊断报告生成”的自定义数据路径
#     report_csv_path: Optional[str] = field(
#         default="/home/lina/RadFM_fine-tuning/Dataset/INbreast/train_and_val_split_radfm_fine-tuning.csv",
#         metadata={"help": "Path to your custom report-generation CSV."}
#     )
#     report_prompt_json: Optional[str] = field(
#         default=None,
#         metadata={"help": "Optional path to a prompt json for report generation; if None, use default report_prompt.json"}
#     )
#
#
# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     remove_unused_columns: bool = field(default=False)
#     batch_size_2D: int = field(default=4)
#     batch_size_3D: int = field(default=1)
#     output_dir: Optional[str] = field(default="/home/lina/RadFM_fine-tuning/Results_train_llama_json")
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#
#     # === LoRA 相关（新增） ===
#     use_lora: bool = field(default=True, metadata={"help": "Whether to inject LoRA into the language model."})
#     lora_r: int = field(default=16)
#     lora_alpha: int = field(default=32)
#     lora_dropout: float = field(default=0.05)
#     lora_target: str = field(default="q_proj,v_proj", metadata={"help": "Comma-separated target module keywords"})
#
#
# @dataclass
# class DataCollator(object):
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         vision_xs, lang_xs, attention_masks, labels, loss_reweight, key_words_query = tuple(
#             [instance[key] for instance in instances] for key in
#             ('vision_x', 'lang_x', 'attention_mask', 'labels', 'loss_reweight', 'key_words_query'))
#
#         lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
#         attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
#         labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
#         loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight], dim=0)
#
#         target_H = 512
#         target_W = 512
#         target_D = 4
#         MAX_D = 0
#
#         D_list = list(range(4, 65, 4))
#         if len(vision_xs) == 1:
#             if vision_xs[0].shape[0] > 6:
#                 D_list = list(range(4, 33, 4))
#
#         for ii in vision_xs:
#             try:
#                 D = ii.shape[-1]
#                 if D > MAX_D:
#                     MAX_D = D
#             except:
#                 continue
#         for temp_D in D_list:
#             if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
#                 target_D = temp_D
#
#         if len(vision_xs) == 1 and target_D > 4:
#             target_H = 256
#             target_W = 256
#
#         vision_xs = [torch.nn.functional.interpolate(s, size=(target_H, target_W, target_D)) for s in vision_xs]
#         vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs, batch_first=True, padding_value=0)
#
#         print(vision_xs.shape, vision_xs.dtype)
#         return dict(
#             lang_x=lang_xs,
#             vision_x=vision_xs,
#             attention_mask=attention_masks,
#             labels=labels,
#             loss_reweight=loss_reweight,
#             key_words_query=key_words_query
#         )
#
#
# def main():
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#
#     # ==== HOTFIX: 兼容自定义 Trainer 期待的字段 ====
#     if not hasattr(training_args, "sharded_ddp"):
#         training_args.sharded_ddp = ""  # Lina added: 让 len(args.sharded_ddp) 不报错
#     # ==============================================
#
#     # Lina added: 解析 lora_target 字符串 → 列表
#     if isinstance(training_args.lora_target, str):
#         training_args.lora_target = [x.strip() for x in training_args.lora_target.split(",") if x.strip()]
#
#     training_args.data_sampler = My_DistributedBatchSampler
#
#     print("Setup Data")
#
#     # original code:
#     # Train_dataset = multi_dataset(text_tokenizer=model_args.tokenizer_path)
#     # Eval_dataset = multi_dataset_close(text_tokenizer=model_args.tokenizer_path)
#     # Lina added: 只加载“诊断报告生成”的自定义 CSV 数据
#     Train_dataset = multi_dataset(
#         text_tokenizer=model_args.tokenizer_path,
#         custom_report_csv=data_args.report_csv_path,
#         custom_prompt_json=data_args.report_prompt_json
#     )
#     Eval_dataset = Train_dataset  # Lina added: 先用同一份数据做验证，保证流程跑通
#
#     print("Setup Model")
#
#     # Lina added: 为 device_map 构造每张卡的显存上限（按需调整）
#     num_gpus = torch.cuda.device_count()
#     max_memory = {i: "45GiB" for i in range(num_gpus)}  # ✅ accelerate 要求用 int 索引
#     # 如需 CPU/DISK offload，也可以加：
#     # max_memory = {i: "45GiB" for i in range(num_gpus)}
#     # max_memory["cpu"] = "128GiB"  # 可选
#     # max_memory["disk"] = "200GiB" # 可选（需传 offload_folder 时）
#
#     # original code:
#     # model = MultiLLaMAForCausalLM(lang_model_path=model_args.lang_encoder_path)
#     model = MultiLLaMAForCausalLM(                              # Lina added: 多卡分片 + FP32
#         lang_model_path=model_args.lang_encoder_path,
#         device_map="auto",
#         max_memory=max_memory,
#     )
#
#     # ---------- 让语言模型词表尺寸与数据侧一致 ----------
#     tok = getattr(Train_dataset, "text_tokenizer", None)  # Lina added
#     print(f"[Tokenizer] len={len(tok) if tok else 'N/A'}, "
#           f"vocab_attr={getattr(Train_dataset, 'voc_size', 'N/A')}")
#     new_vocab = (len(tok) if tok is not None else getattr(Train_dataset, "voc_size", None))  # Lina added
#     old_vocab = model.lang_model.get_output_embeddings().weight.size(0)
#     if new_vocab is not None and new_vocab != old_vocab:
#         print(f"[Vocab] resize LM embeddings: {old_vocab} -> {new_vocab}")
#         model.lang_model.resize_token_embeddings(new_vocab)              # Lina added
#         if hasattr(model, "voc_size"):
#             model.voc_size = new_vocab                                   # Lina added
#         # 关键：resize 后重绑 RadFM 的自定义嵌入
#         model.embedding_layer.weight = model.lang_model.get_input_embeddings().weight  # Lina added
#     # ----------------------------------------------------
#
#     # --- Sanity checks ---
#     print("LM handle exists? ", hasattr(model, "lang_model"))
#     print("LM class: ", type(model.lang_model))
#     count = 0
#     for n, _ in model.lang_model.named_modules():
#         if n.endswith(("q_proj", "v_proj")):
#             print("[targetable]", n)
#             count += 1
#             if count >= 8:
#                 break
#     vocab = model.lang_model.get_output_embeddings().weight.size(0)
#     print("LM vocab size (from head):", vocab)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"Total params (pre-LoRA): {total / 1e6:.2f}M")
#
#     # === LoRA 注入（只打语言模型）===
#     if training_args.use_lora:
#         # 冻结视觉/嵌入侧（建议）
#         if hasattr(model, "embedding_layer"):
#             for n, p in model.embedding_layer.named_parameters():
#                 p.requires_grad = False
#
#         lora_cfg = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             r=training_args.lora_r,
#             lora_alpha=training_args.lora_alpha,
#             lora_dropout=training_args.lora_dropout,
#             bias="none",
#             target_modules=training_args.lora_target,
#         )
#
#         lang_model = getattr(model, "lang_model", None)
#         if lang_model is None:
#             raise ValueError(
#                 "未在 MultiLLaMAForCausalLM 中找到 `lang_model`。"
#                 "请在 src/Model/RadFM/multimodality_model.py 的 __init__ 里确保暴露：self.lang_model = <LlamaForCausalLM实例>"
#             )
#
#         lang_model = get_peft_model(lang_model, lora_cfg)  # Lina added
#         setattr(model, "lang_model", lang_model)           # Lina added
#
#         trainable, total = 0, 0
#         for n, p in model.named_parameters():
#             total += p.numel()
#             if p.requires_grad:
#                 trainable += p.numel()
#         print(f"[LoRA] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")
#
#     trainer = Trainer(model=model,
#                       train_dataset=Train_dataset,
#                       eval_dataset=Eval_dataset,
#                       args=training_args,
#                       data_collator=DataCollator(),
#                       compute_metrics=compute_metrics
#                       )
#
#     trainer.train()
#     trainer.save_state()
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os       # Lina added
import sys      # Lina added

DEFAULT_GPUS = "0,1,2,3,5,6"  # Lina added
_cli_gpus = None              # Lina added
for i, a in enumerate(sys.argv):  # Lina added
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
# ------------------------------------------------------------

import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Dataset.multi_dataset import multi_dataset
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
import numpy as np
import torch

# === 新增：PEFT / LoRA ===
from peft import LoraConfig, get_peft_model, TaskType  # Lina added

# === 新增：Accelerate 显式分片加载 ===
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, LlamaForCausalLM


def compute_metrics(eval_preds):
    ACCs = eval_preds.predictions
    return {"accuracy": np.mean(ACCs, axis=-1)}


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/home/lina/GitHub/Llama-2-13b-hf")
    tokenizer_path: str = field(default='/home/lina/GitHub/Llama-2-13b-hf',
                                metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    # Lina added: 只跑“诊断报告生成”的自定义数据路径
    report_csv_path: Optional[str] = field(
        default="/home/lina/RadFM_fine-tuning/Dataset/INbreast/train_and_val_split_radfm_fine-tuning.csv",
        metadata={"help": "Path to your custom report-generation CSV."}
    )
    report_prompt_json: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a prompt json for report generation; if None, use default report_prompt.json"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(default="/home/lina/RadFM_fine-tuning/Results_train_llama_json")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    # === 显存友好项（新增，默认开启） ===
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)

    # === LoRA 相关（新增） ===
    use_lora: bool = field(default=True, metadata={"help": "Whether to inject LoRA into the language model."})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target: str = field(default="q_proj,v_proj", metadata={"help": "Comma-separated target module keywords"})


@dataclass
class DataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vision_xs, lang_xs, attention_masks, labels, loss_reweight, key_words_query = tuple(
            [instance[key] for instance in instances] for key in
            ('vision_x', 'lang_x', 'attention_mask', 'labels', 'loss_reweight', 'key_words_query'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight], dim=0)

        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except Exception:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        # === 限制体素深度与分辨率，缓解 OOM（可按需调整） ===
        target_D = min(target_D, 16)
        if len(vision_xs) > 1:
            target_H = 256
            target_W = 256

        vision_xs = [torch.nn.functional.interpolate(s, size=(target_H, target_W, target_D)) for s in vision_xs]
        vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs, batch_first=True, padding_value=0)

        # === 关键：视觉张量降为半精度，省显存 ===
        vision_xs = vision_xs.to(dtype=torch.float16)

        print(vision_xs.shape, vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_masks,
            labels=labels,
            loss_reweight=loss_reweight,
            key_words_query=key_words_query
        )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ==== HOTFIX: 兼容自定义 Trainer 期待的字段 ====
    if not hasattr(training_args, "sharded_ddp"):
        training_args.sharded_ddp = ""  # Lina added: 让 len(args.sharded_ddp) 不报错
    # ==============================================

    # Lina added: 解析 lora_target 字符串 → 列表
    if isinstance(training_args.lora_target, str):
        training_args.lora_target = [x.strip() for x in training_args.lora_target.split(",") if x.strip()]

    training_args.data_sampler = My_DistributedBatchSampler

    print("Setup Data")

    # 只加载“诊断报告生成”的自定义 CSV 数据
    Train_dataset = multi_dataset(
        text_tokenizer=model_args.tokenizer_path,
        custom_report_csv=data_args.report_csv_path,
        custom_prompt_json=data_args.report_prompt_json
    )
    Eval_dataset = Train_dataset  # 先用同一份数据做验证，保证流程跑通

    print("Setup Model")

    # 为多卡分片构造每张卡的显存上限
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "45GiB" for i in range(num_gpus)}  # accelerate 期待 int 索引
    # 如需 CPU/DISK offload，可按需添加：
    # max_memory["cpu"] = "128GiB"
    # max_memory["disk"] = "200GiB"

    # === 1) 先构建多模态外壳。尽量避免其内部直接 from_pretrained 占显存 ===
    # 如果你的 MultiLLaMAForCausalLM 允许 lang_model_path=None，则用 None；
    # 若不允许，可暂时传入原路径，随后立刻覆盖其 lang_model（见下）。
    try:
        model = MultiLLaMAForCausalLM(
            lang_model_path=None  # 推荐：让我们手动挂载 lang_model
        )
    except Exception:
        # 兼容：有些实现必须提供路径
        model = MultiLLaMAForCausalLM(
            lang_model_path=model_args.lang_encoder_path
        )
        # 若它已把大模型加载到了 GPU，这一步可能已经吃显存；我们会马上覆盖其 lang_model。

    # === 2) 用 Accelerate 三步法分片加载 LLaMA ===
    cfg = AutoConfig.from_pretrained(model_args.lang_encoder_path)

    # 空权重构图（不占显存）
    with init_empty_weights():
        lm = LlamaForCausalLM.from_config(cfg)

    # 推断设备映射（结合 max_memory）
    try:
        device_map = infer_auto_device_map(
            lm,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
        )
    except Exception:
        # 部分旧版 accelerate 无 infer_auto_device_map；退化为均匀切分（简单策略）
        # 你也可以在这里手写一个 device_map 字典
        device_map = "auto"

    # 分片加载权重到多卡
    lm = load_checkpoint_and_dispatch(
        lm,
        checkpoint=model_args.lang_encoder_path,   # 目录（含 *.bin.index.json）或单文件 *.bin
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32),
        offload_folder=None,
    )

    # 训练友好项：关闭 cache + 开梯度检查点（若开启）
    lm.config.use_cache = False
    if getattr(training_args, "gradient_checkpointing", False) and hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()

    # 把分片后的 LLM 挂回多模态模型
    setattr(model, "lang_model", lm)

    # 可选：把视觉分支放到最后一张卡，避免 cuda:0 过载
    if torch.cuda.device_count() > 1 and hasattr(model, "embedding_layer"):
        try:
            model.embedding_layer.to(f"cuda:{torch.cuda.device_count()-1}")
        except Exception:
            pass

    # ---------- 让语言模型词表尺寸与数据侧一致 ----------
    tok = getattr(Train_dataset, "text_tokenizer", None)
    print(f"[Tokenizer] len={len(tok) if tok else 'N/A'}, "
          f"vocab_attr={getattr(Train_dataset, 'voc_size', 'N/A')}")
    new_vocab = (len(tok) if tok is not None else getattr(Train_dataset, "voc_size", None))
    old_vocab = model.lang_model.get_output_embeddings().weight.size(0)
    if new_vocab is not None and new_vocab != old_vocab:
        print(f"[Vocab] resize LM embeddings: {old_vocab} -> {new_vocab}")
        model.lang_model.resize_token_embeddings(new_vocab)
        if hasattr(model, "voc_size"):
            model.voc_size = new_vocab
        # 关键：resize 后重绑 RadFM 的自定义嵌入
        if hasattr(model, "embedding_layer"):
            try:
                model.embedding_layer.weight = model.lang_model.get_input_embeddings().weight
            except Exception:
                pass
    # ----------------------------------------------------

    # --- Sanity checks ---
    print("LM handle exists? ", hasattr(model, "lang_model"))
    print("LM class: ", type(model.lang_model))
    # 打印 hf_device_map，确认确实多卡分片
    print("hf_device_map =", getattr(model.lang_model, "hf_device_map", None))

    count = 0
    for n, _ in model.lang_model.named_modules():
        if n.endswith(("q_proj", "v_proj")):
            print("[targetable]", n)
            count += 1
            if count >= 8:
                break
    vocab = model.lang_model.get_output_embeddings().weight.size(0)
    print("LM vocab size (from head):", vocab)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params (pre-LoRA): {total / 1e6:.2f}M")

    # === LoRA 注入（只打语言模型）===
    if training_args.use_lora:
        # 冻结视觉/嵌入侧（建议）
        if hasattr(model, "embedding_layer"):
            for n, p in model.embedding_layer.named_parameters():
                p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            target_modules=training_args.lora_target,
        )

        lang_model = getattr(model, "lang_model", None)
        if lang_model is None:
            raise ValueError(
                "未在 MultiLLaMAForCausalLM 中找到 `lang_model`。"
                "请在 src/Model/RadFM/multimodality_model.py 的 __init__ 里确保暴露：self.lang_model = <LlamaForCausalLM实例>"
            )

        lang_model = get_peft_model(lang_model, lora_cfg)  # Lina added
        setattr(model, "lang_model", lang_model)           # Lina added

        trainable, total = 0, 0
        for n, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"[LoRA] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    # 训练前再保险：确保不使用 KV cache
    if hasattr(model, "lang_model") and hasattr(model.lang_model, "config"):
        model.lang_model.config.use_cache = False

    trainer = Trainer(model=model,
                      train_dataset=Train_dataset,
                      eval_dataset=Eval_dataset,
                      args=training_args,
                      data_collator=DataCollator(),
                      compute_metrics=compute_metrics
                      )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
