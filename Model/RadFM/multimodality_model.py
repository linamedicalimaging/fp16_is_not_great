from torch import nn
from transformers.models.llama import LlamaForCausalLM
from transformers import AutoConfig
from .my_embedding_layer import MyEmbedding
from torch.nn import CrossEntropyLoss
import tqdm.auto as tqdm
import torch
import re
import math

loss_fct = CrossEntropyLoss(reduction='none')

_QKVO_RE = re.compile(
    r"^model\.layers\.(\d?:[0-9]|1[0-9])\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")

# _QKVO_RE = re.compile(
#     r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")

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

class MultiLLaMAForCausalLM(nn.Module):
    def __init__(self, lang_model_path):
        super(MultiLLaMAForCausalLM, self).__init__()  
        config = AutoConfig.from_pretrained(lang_model_path)
        self.lang_model = LlamaForCausalLM(config)
        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.hidden_dim = 5120
        self.voc_size = 32000

    def add_freeze_non_lora(self, rank, alpha, dropout):
        self.lang_model = apply_lora_to_model(self.lang_model, rank=rank, alpha=alpha, dropout=dropout)
        freeze_non_lora(self.lang_model)
        freeze_non_lora(self.embedding_layer)

    def forward(self,lang_x, vision_x, attention_mask, labels, loss_reweight,key_words_query):
        if labels.shape == lang_x.shape:
            self.embedding_layer.flag = 'Text'
            input_embedding,loss_match= self.embedding_layer(lang_x, vision_x,key_words_query)   # ,loss_matching
            output = self.lang_model(inputs_embeds = input_embedding,attention_mask = attention_mask, labels = labels)
            logits = output['logits']

            loss_reg = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[...,1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(shift_logits.device)
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight*loss_reg)/torch.sum(shift_loss_reweight)
            loss = loss_reg
            if loss_match!= None:
                loss = 0.8*loss + 0.2*loss_match
            return loss

        ### useless for now ignore the folowing codes ###
        # if labels.shape == vision_x.shape:
        #    self.embedding_layer.flag = 'Seg'
        #    input_embedding = self.embedding_layer(lang_x, vision_x)
    
    def generate(self, lang_x,vision_x):
        self.embedding_layer.flag = 'Text'
        with torch.no_grad():
            input_embedding,_ = self.embedding_layer(lang_x, vision_x) 
            generation = self.lang_model.generate(inputs_embeds = input_embedding, max_new_tokens =200,top_k=50)
        return generation
