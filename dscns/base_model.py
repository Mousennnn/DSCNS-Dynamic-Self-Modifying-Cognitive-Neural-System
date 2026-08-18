"""Shared base language model (Phase 0 output).

A frozen GPT-2-class model that provides:
- mean-pooled embeddings for knowledge items,
- CE-loss / uncertainty proxies,
- generation for answer-based evaluation,
and hosts one PeftModel with one LoRA adapter per cognitive network so that
all networks share the base weights while keeping local parameter spaces
Theta_i (memory-efficient multi-adapter setup).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class BaseLanguageModel:
    """Frozen base model + multi-adapter PeftModel wrapper."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda",
                 torch_dtype: str = "float32", max_len: int = 256,
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
                 cache_dir: Optional[str] = None):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float32
        self.dtype = dtype
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.hidden_dim = self.model.config.n_embd

        # first adapter (created via get_peft_model)
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"], bias="none", task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.model, lora_config, adapter_name="base")
        self.peft_model.eval()

        self._adapter_names = ["base"]
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    # ------------------------------------------------------------------ #
    def add_adapter(self, adapter_name: str) -> None:
        """Register a new LoRA adapter for a cognitive network."""
        from peft import LoraConfig

        if adapter_name in self._adapter_names:
            return
        config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
            target_modules=["c_attn", "c_proj"], bias="none", task_type="CAUSAL_LM",
        )
        try:
            # peft >= 0.10: add fresh adapter from config
            self.peft_model.add_adapter(adapter_name, config)
        except TypeError:
            self.peft_model.load_adapter(config, adapter_name=adapter_name)
        self._adapter_names.append(adapter_name)
        self.peft_model.set_adapter("base")

    def set_active_adapter(self, adapter_name: str) -> None:
        self.peft_model.set_adapter(adapter_name)

    # ------------------------------------------------------------------ #
    def embed_and_loss(self, texts: List[str], max_len: Optional[int] = None,
                       batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Mean-pooled last-hidden embeddings and mean CE losses (frozen base)."""
        import torch

        max_len = max_len or self.max_len
        embs: List[np.ndarray] = []
        losses: List[float] = []
        self.set_active_adapter("base")
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_len)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc, labels=enc["input_ids"],
                                 output_hidden_states=True)
                loss = out.loss
                hidden = out.hidden_states[-1]  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                embs.append(pooled.float().cpu().numpy())
                losses.extend([float(loss.item())] * len(chunk))
                del out, hidden, pooled, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(embs, axis=0), np.asarray(losses, dtype=np.float32)

    def embed(self, texts: List[str], max_len: Optional[int] = None,
              batch_size: int = 8) -> np.ndarray:
        embs, _ = self.embed_and_loss(texts, max_len=max_len, batch_size=batch_size)
        return embs

    def losses(self, texts: List[str], max_len: Optional[int] = None,
               batch_size: int = 8) -> np.ndarray:
        _, losses = self.embed_and_loss(texts, max_len=max_len, batch_size=batch_size)
        return losses

    # ------------------------------------------------------------------ #
    def generate(self, prompts: List[str], max_new_tokens: int = 48,
                 batch_size: int = 8) -> List[str]:
        """Greedy generation with the currently active adapter."""
        import torch

        out_texts = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_len)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.peft_model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
            for g in gen:
                text = self.tokenizer.decode(
                    g[enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                out_texts.append(text.strip())
        return out_texts

    # ------------------------------------------------------------------ #
    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.peft_model.parameters()
                       if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
