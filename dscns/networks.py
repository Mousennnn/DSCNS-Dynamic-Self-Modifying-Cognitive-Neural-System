"""Multi-network cognitive layer (report section 2.1 / 3.3).

Each CognitiveNetwork wraps a shared base model with its own LoRA adapter
(its local parameter space Theta_i), keeps a domain embedding, tracks the
per-knowledge internalization level I_ij and knowledge state level (0..3),
and implements the four-dimensional evaluation Q_i(K) = (R, N, C, I).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .utils import cosine_similarity, cosine_matrix, loss_to_confidence


class CognitiveNetwork:
    """A single cognitive network N_i."""

    def __init__(
        self,
        net_id: str,
        name: str,
        domain: str,
        peft_model: Any,
        memory: Any,
        domain_embedding: Optional[np.ndarray] = None,
        base_lr: float = 5e-4,
        source_weight: float = 0.5,
        trust: float = 0.5,
        plasticity: Any = None,          # Phase 5: IntrinsicPlasticityModule
        plasticity_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.id = net_id
        self.name = name
        self.domain = domain
        self.data_domain = None  # matching experience-stream domain (e.g. 'math')
        self.peft_model = peft_model  # shared multi-adapter PeftModel
        self.memory = memory
        self.domain_embedding = domain_embedding
        self.base_lr = base_lr
        self.source_weight = source_weight
        self.trust = trust  # initial trust weight, updated by verification net

        # knowledge bookkeeping
        self.internalization_level: Dict[str, float] = {}   # knowledge_id -> I_ij
        self.knowledge_states: Dict[str, int] = {}          # knowledge_id -> Level 0..3
        self.accepted_embeddings: List[np.ndarray] = []
        self.accepted_ids: List[str] = []
        self.recent_domains: List[str] = []                 # task diversity tracking
        self.activation_count: int = 0

        # performance tracking
        self.performance_history: List[float] = []
        self.baseline_performance: float = 0.0
        self.competence: float = 0.0
        self.uncertainty: float = 0.5
        self.lr = base_lr

        # message handling (communication bus)
        self.inbox: List[Any] = []
        self.corrections_received: int = 0
        self.queries_answered: int = 0

        # optimizer over this network's adapter parameters
        self.optimizer = None

        # Phase 5: intrinsic parameter self-modification
        self.plasticity = plasticity          # IntrinsicPlasticityModule or None
        self.plasticity_cfg = plasticity_cfg or {}
        self.plasticity_optimizer = None      # used by P5-C offline training
        self.step_count: int = 0              # grad-step counter (external trigger)
        self.modification_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # adapter management
    # ------------------------------------------------------------------ #
    def _adapter_params(self) -> List[Any]:
        """Parameters belonging to this network's LoRA adapter."""
        params = []
        for n, p in self.peft_model.named_parameters():
            if ("lora" in n) and (f".{self.id}." in n or n.endswith(f".{self.id}")):
                params.append(p)
        return params

    def set_trainable(self, trainable: bool = True) -> None:
        for n, p in self.peft_model.named_parameters():
            if "lora" in n:
                p.requires_grad = (trainable and
                                   (f".{self.id}." in n or n.endswith(f".{self.id}")))
            else:
                p.requires_grad = False

    def get_optimizer(self):
        if self.optimizer is None:
            import torch

            params = [p for p in self._adapter_params() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(params, lr=self.base_lr, weight_decay=0.0)
        return self.optimizer

    def snapshot_adapter(self) -> Dict[str, Any]:
        import torch

        return {n: p.detach().clone() for n, p in self.peft_model.named_parameters()
                if ("lora" in n) and (f".{self.id}." in n or n.endswith(f".{self.id}"))}

    def restore_adapter(self, snap: Dict[str, Any]) -> None:
        with_ = self.peft_model.named_parameters()
        for n, p in with_:
            if n in snap:
                p.data.copy_(snap[n])

    # ------------------------------------------------------------------ #
    # four-dimensional evaluation  Q_i(K) = (R, N, C, I)
    # ------------------------------------------------------------------ #
    def evaluate(self, candidate: Any) -> Dict[str, float]:
        """Independent evaluation of candidate knowledge K by this network."""
        # 1. Relevance: similarity to this network's domain embedding
        #    (+ a domain-match bonus grounding the stream's domain labels)
        R = 0.0
        if self.domain_embedding is not None and candidate.embedding is not None:
            R = cosine_similarity(candidate.embedding, self.domain_embedding)
        if self.data_domain is not None and candidate.domain == self.data_domain:
            R = float(np.clip(R + 0.25, 0.0, 1.0))

        # 2. Novelty: 1 - max similarity against already-internalized knowledge
        N = 1.0
        if self.accepted_embeddings and candidate.embedding is not None:
            sims = cosine_matrix(
                np.asarray(candidate.embedding)[None, :],
                np.stack(self.accepted_embeddings),
            )[0]
            N = float(np.clip(1.0 - float(sims.max()), 0.0, 1.0))

        # 3. Confidence: base-model confidence x source reliability
        #    (relative calibration: base GPT-2 CE on raw text ~ 4-5 nats)
        base_conf = float(np.clip(1.0 - candidate.loss / 5.0, 0.0, 1.0))
        C = float(np.clip(base_conf * (self.source_weight +
                                        (1 - self.source_weight) * candidate.source_reliability),
                          0.0, 1.0))

        # 4. Importance: relevance x uncertainty-driven utility
        I = float(np.clip(R * (0.3 + 0.7 * candidate.uncertainty), 0.0, 1.0))

        return {"R": R, "N": N, "C": C, "I": I}

    # ------------------------------------------------------------------ #
    # knowledge state levels (report section 3.2)
    # ------------------------------------------------------------------ #
    def observe(self, candidate: Any) -> None:
        """Level 1: existence-level cognition (receiving information)."""
        self.knowledge_states.setdefault(candidate.id, 1)
        self.recent_domains.append(candidate.domain)
        self.recent_domains = self.recent_domains[-200:]
        self.activation_count += 1

    def store_as_callable(self, candidate: Any) -> None:
        """Level 2: callable cognition (queryable from other networks)."""
        self.knowledge_states[candidate.id] = max(
            self.knowledge_states.get(candidate.id, 1), 2
        )

    def mark_internalized(self, candidate: Any, level: float) -> None:
        """Level 3: internally internalized (local neural representation)."""
        self.knowledge_states[candidate.id] = 3
        self.internalization_level[candidate.id] = float(level)
        if candidate.embedding is not None:
            self.accepted_embeddings.append(np.asarray(candidate.embedding, dtype=np.float32))
            self.accepted_ids.append(candidate.id)
        if candidate.id not in self.internalization_level:
            self.internalization_level[candidate.id] = float(level)

    # ------------------------------------------------------------------ #
    # learning primitives used by the InternalizationController
    # ------------------------------------------------------------------ #
    def compute_trial_update(self, texts: List[str], tokenizer: Any,
                             alpha: float, batch_size: int = 4,
                             max_len: int = 256) -> Any:
        """One small exploratory optimizer step (theta + alpha*dtheta)."""
        import torch

        self.set_trainable(True)
        self.peft_model.set_adapter(self.id)
        self.peft_model.train()
        opt = self.get_optimizer()
        opt.zero_grad()

        lr_scale = max(alpha, 1e-4)
        for g in opt.param_groups:
            g["lr"] = self.base_lr * lr_scale

        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len,
        )
        enc = {k: v.to(self.peft_model.device) for k, v in enc.items()}
        loss_sum = 0.0
        n_batches = 0
        for i in range(0, len(texts), batch_size):
            chunk = {k: v[i:i + batch_size] for k, v in enc.items()}
            out = self.peft_model(**chunk, labels=chunk["input_ids"])
            loss = out.loss / max(1, (len(texts) + batch_size - 1) // batch_size)
            loss.backward()
            loss_sum += float(out.loss.item())
            n_batches += 1
        torch.nn.utils.clip_grad_norm_([p for p in self._adapter_params()
                                        if p.grad is not None], max_norm=1.0)
        opt.step()
        self.peft_model.eval()
        return {"trial_loss": loss_sum / max(n_batches, 1)}

    def apply_update(self) -> None:
        """Consolidate the accepted update (no-op; weights already stepped)."""
        self.set_trainable(False)

    def rollback(self, snapshot: Dict[str, Any]) -> None:
        """Restore pre-trial adapter weights (and reset optimizer state)."""
        self.restore_adapter(snapshot)
        self.optimizer = None  # drop stale moment estimates from the trial
        self.set_trainable(False)

    # ------------------------------------------------------------------ #
    # performance evaluation
    # ------------------------------------------------------------------ #
    def evaluate_texts(self, texts: List[str], tokenizer: Any,
                       batch_size: int = 8, max_len: int = 256) -> float:
        """Mean exp(-CE-loss) over texts with THIS network's adapter active."""
        losses = self.losses_for_texts(texts, tokenizer, batch_size, max_len)
        if not losses.size:
            return 0.0
        return float(np.exp(-float(np.mean(losses))))

    def losses_for_texts(self, texts: List[str], tokenizer: Any,
                         batch_size: int = 8, max_len: int = 256) -> np.ndarray:
        """Per-text mean CE losses with THIS network's adapter active."""
        import torch

        if not texts:
            return np.zeros(0, dtype=np.float32)
        self.peft_model.set_adapter(self.id)
        self.peft_model.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = tokenizer(chunk, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_len)
                enc = {k: v.to(self.peft_model.device) for k, v in enc.items()}
                out = self.peft_model(**enc, labels=enc["input_ids"])
                shift_logits = out.logits[:, :-1, :].float()
                shift_labels = enc["input_ids"][:, 1:]
                shift_mask = enc["attention_mask"][:, 1:]
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1), reduction="none",
                ).reshape(shift_logits.size(0), -1)
                per = (ce * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
                losses.extend(per.float().cpu().numpy().tolist())
                del out, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.asarray(losses, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # message handling (report section 8.1)
    # ------------------------------------------------------------------ #
    async def receive(self, message: Any) -> None:
        """Asynchronous message reception (communication bus)."""
        from .communication import MessageType

        self.inbox.append(message)
        mtype = message.msg_type
        if mtype == MessageType.BROADCAST:
            cand = message.content.get("candidate")
            if cand is not None:
                self.observe(cand)
        elif mtype == MessageType.QUERY:
            self.queries_answered += 1
        elif mtype == MessageType.CONFLICT:
            # lower our stated confidence on the disputed item
            self.corrections_received += 1
        elif mtype == MessageType.CORRECTION:
            self.corrections_received += 1
        elif mtype == MessageType.UPDATE_NOTIFY:
            pass
        elif mtype == MessageType.MERGE_REQUEST:
            pass
        elif mtype == MessageType.SPLIT_NOTIFY:
            pass
        elif mtype == MessageType.META_REPORT:
            pass

    def deliver(self, message: Any) -> None:
        """Synchronous delivery used by the drain loop."""
        self.inbox.append(message)

    # ------------------------------------------------------------------ #
    # structure-evolution statistics (report section 4)
    # ------------------------------------------------------------------ #
    def task_diversity(self) -> float:
        from .utils import entropy

        if not self.recent_domains:
            return 0.0
        counts: Dict[str, int] = {}
        for d in self.recent_domains:
            counts[d] = counts.get(d, 0) + 1
        probs = [c / len(self.recent_domains) for c in counts.values()]
        return entropy(probs) / np.log(max(len(counts), 2))

    def representation_embedding(self) -> np.ndarray:
        """Mean embedding of internalized knowledge (representation centroid)."""
        if not self.accepted_embeddings:
            return np.zeros(1, dtype=np.float32)
        return np.mean(np.stack(self.accepted_embeddings), axis=0)

    # ------------------------------------------------------------------ #
    # Phase 5: intrinsic parameter self-modification
    #   theta -> h -> delta_theta -> theta'   (report sections 4-7)
    #
    # Plasticity *generation* (P_phi(h, stats(theta), s)) and the parameter
    # *transition* (theta' = theta + alpha*delta) are model-side mechanisms;
    # triggers, validation and rollback are experiment-controller concerns.
    # ------------------------------------------------------------------ #
    def _current_params_tensors(self) -> Dict[str, Any]:
        """Flattened live copies of this adapter's lora_A / lora_B weights.

        Used only for the low-dimensional stats(theta) input of P_phi;
        the full parameter vector is never passed to the plasticity module.
        """
        import torch

        wa, wb = [], []
        for n, p in self.peft_model.named_parameters():
            if f".{self.id}." in n:
                if "lora_A" in n:
                    wa.append(p.detach().flatten())
                elif "lora_B" in n:
                    wb.append(p.detach().flatten())
        W_A = torch.cat(wa) if wa else torch.zeros(1)
        W_B = torch.cat(wb) if wb else torch.zeros(1)
        return {"W_A": W_A, "W_B": W_B}

    def param_stats(self) -> List[float]:
        """[mean, std, min, max] over this network's adapter parameters."""
        import torch

        tensors = self._current_params_tensors()
        all_ = torch.cat([tensors["W_A"], tensors["W_B"]])
        return [float(all_.mean()), float(all_.std()),
                float(all_.min()), float(all_.max())]

    def _get_meta_info(self, meta_dim: int = 32) -> Any:
        """Build the self-state meta vector s_t (padded to meta_dim).

        Combines the Phase 4 self-state features with learning-progress,
        modification count and step count.  This is model-side meta info,
        not an external observation.
        """
        import torch

        s = self.get_self_state()
        hist = self.performance_history
        progress = float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        feats = [
            s["competence"], s["uncertainty"], progress,
            float(len(self.modification_history)), float(self.step_count),
            s["task_diversity"], s["trust"], s["log_accepted"],
            s["activation_norm"], s["perf_trend"], s["queries_norm"],
            s["corrections_norm"], s["bookkeeping_norm"],
        ]
        vec = torch.tensor(feats, dtype=torch.float32)
        if vec.size(0) < meta_dim:
            vec = torch.cat([vec, torch.zeros(meta_dim - vec.size(0))])
        return vec[:meta_dim]

    def snapshot_parameters(self) -> Dict[str, Any]:
        """Snapshot adapter weights (experiment safety mechanism, section 7.2)."""
        return {
            "lora_A": {n: p.detach().clone()
                       for n, p in self.peft_model.named_parameters()
                       if f".{self.id}." in n and "lora_A" in n},
            "lora_B": {n: p.detach().clone()
                       for n, p in self.peft_model.named_parameters()
                       if f".{self.id}." in n and "lora_B" in n},
            "step": self.step_count,
        }

    def restore_parameters(self, snapshot: Dict[str, Any]) -> None:
        """Restore adapter weights from a snapshot (rollback safety)."""
        with self._no_grad_ctx():
            for n, p in self.peft_model.named_parameters():
                if n in snapshot.get("lora_A", {}):
                    p.data.copy_(snapshot["lora_A"][n])
                elif n in snapshot.get("lora_B", {}):
                    p.data.copy_(snapshot["lora_B"][n])

    @staticmethod
    def _no_grad_ctx():
        import torch

        return torch.no_grad()

    def apply_intrinsic_modification(self, delta_params: Dict[str, Any],
                                     alpha: float = 1.0,
                                     permanent: bool = True) -> bool:
        """Model-side parameter transition: theta' = theta + alpha * delta.

        delta_params: {'delta_W_A': (H, r), 'delta_W_B': (r, H), ...}
        The same low-rank delta is applied to every LoRA injection point of
        this network's adapter (all lora_A/lora_B matrices share shape
        (r, H) / (H, r) in this prototype).
        """
        dA = delta_params["delta_W_A"].transpose(0, 1)  # (r, H)
        dB = delta_params["delta_W_B"].transpose(0, 1)  # (H, r)
        with self._no_grad_ctx():
            for n, p in self.peft_model.named_parameters():
                if f".{self.id}." in n:
                    if "lora_A" in n and p.size(1) == dA.size(1):
                        # down-projections with hidden-size input: c_attn
                        # and attention c_proj (mlp c_proj input is 3072)
                        p.data.add_(dA.to(p.device) * alpha)
                    elif "lora_B" in n and p.size(0) == dB.size(0):
                        # up-projections with hidden-size output; c_attn's
                        # (3*H, r) QKV output is skipped in this prototype
                        p.data.add_(dB.to(p.device) * alpha)
        self.modification_history.append({
            "step": self.step_count,
            "delta_W_A_norm": float(delta_params["delta_W_A"].norm()),
            "delta_W_B_norm": float(delta_params["delta_W_B"].norm()),
            "modulation_strength": float(delta_params.get("modulation_strength", 0.0)),
            "alpha": alpha,
            "permanent": permanent,
        })
        return True

    def generate_delta(self, texts: List[str], tokenizer: Any,
                       meta_info: Optional[Any] = None,
                       batch_size: int = 8, max_len: int = 192,
                       grad_enabled: bool = False) -> Dict[str, Any]:
        """Model-side plasticity generation: P_phi(h, stats(theta), s).

        Runs the base model + this adapter on ``texts``, pools the last
        hidden states, encodes current parameter statistics and the meta
        vector, and returns the generated delta (consensus over the batch).

        ``grad_enabled`` (P5-C training) keeps gradients flowing through the
        plasticity module only: the base-model forward and the parameter
        statistics stay detached, so the frozen base never receives grads.
        """
        import torch

        if self.plasticity is None:
            raise RuntimeError("network has no plasticity module")
        self.peft_model.set_adapter(self.id)
        self.peft_model.eval()
        self.plasticity.eval()
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        enc = {k: v.to(self.peft_model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.peft_model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # (B, T, H), detached
            B = hidden.size(0)
            if meta_info is None:
                meta_dim = self.plasticity_cfg.get("meta_dim", 32)
                meta_info = self._get_meta_info(meta_dim)
                meta_info = meta_info.to(hidden.device).unsqueeze(0).expand(B, -1)
            current_params = self._current_params_tensors()
        if grad_enabled:
            self.plasticity.train()
            delta = self.plasticity(
                hidden, current_params, meta_info,
                mask=enc["attention_mask"],
            )
        else:
            with torch.no_grad():
                delta = self.plasticity(
                    hidden, current_params, meta_info,
                    mask=enc["attention_mask"],
                )
        delta["meta_info"] = meta_info.detach()
        return delta

    def modulate_forward(self, texts: List[str], tokenizer: Any,
                         delta_params: Dict[str, Any], alpha: float = 1.0,
                         max_len: int = 192) -> Dict[str, Any]:
        """P5-A parameter *modulation*: apply delta transiently, then restore.

        Verifies that internal state can influence the model's own
        computation without permanently modifying parameters.
        """
        import torch

        snap = self.snapshot_parameters()
        logits_before = self._logits_for_texts(texts, tokenizer, max_len)
        self.apply_intrinsic_modification(delta_params, alpha=alpha)
        logits_after = self._logits_for_texts(texts, tokenizer, max_len)
        self.restore_parameters(snap)
        return {
            "logits_before": logits_before,
            "logits_after": logits_after,
            "logits_diff": float((logits_after - logits_before).abs().mean()),
            "weights_restored": True,
        }

    def _logits_for_texts(self, texts: List[str], tokenizer: Any,
                          max_len: int = 192) -> Any:
        import torch

        self.peft_model.set_adapter(self.id)
        self.peft_model.eval()
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        enc = {k: v.to(self.peft_model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.peft_model(**enc)
            return out.logits.detach()

    # ------------------------------------------------------------------ #
    # Phase 5.1: self-modification with magnitude and target (report §5-9)
    # ------------------------------------------------------------------ #
    def apply_self_modification(self, proposal: Any, alpha: float = 1.0) -> bool:
        """Apply a ModificationProposal (magnitude + target selection).

        proposal: dict with delta_W_A, delta_W_B, magnitude, target_group.
        The magnitude scales the update and the target_group selects
        which adapter projection group receives the delta.

        If proposal contains 'alpha_override', that value is used instead
        of the default alpha (for failure injection experiments).

        Returns True if at least one parameter was modified.
        """
        import torch

        dA = proposal["delta_W_A"].transpose(0, 1)   # (r, H)
        dB = proposal["delta_W_B"].transpose(0, 1)   # (H, r)
        magnitude = float(proposal["magnitude"])
        target = int(proposal["target_group"])
        use_alpha = proposal.get("alpha_override", alpha)
        effective_alpha = use_alpha * magnitude
        modified = False
        with torch.no_grad():
            for n, p in self.peft_model.named_parameters():
                if f".{self.id}." not in n:
                    continue
                is_attn = ("h." in n and "attn" in n) or ("attn" in n and "transformer" in n)
                is_mlp = ("mlp" in n or ("h." in n and "mlp" in n))
                if "lora_A" in n and target == 0 and p.size(1) == dA.size(1):
                    # attn lora_A: target group 0
                    p.data.add_(dA.to(p.device) * effective_alpha)
                    modified = True
                elif "lora_B" in n and p.size(0) == dB.size(0):
                    if target == 1 and is_attn:
                        p.data.add_(dB.to(p.device) * effective_alpha)
                        modified = True
                    elif target == 2 and is_mlp:
                        p.data.add_(dB.to(p.device) * effective_alpha)
                        modified = True
        self.modification_history.append({
            "step": self.step_count,
            "magnitude": magnitude,
            "target_group": target,
            "effective_alpha": effective_alpha,
            "delta_W_A_norm": float(proposal["delta_W_A"].norm()),
            "delta_W_B_norm": float(proposal["delta_W_B"].norm()),
            "confidence": float(proposal.get("confidence", 0)),
        })
        return modified


    # ------------------------------------------------------------------ #
    # Phase 4: model self-state interface (proposal section 15)
    # ------------------------------------------------------------------ #
    def get_self_state(self) -> Dict[str, float]:
        """Per-network self-state features used by the learned policy.

        Collected together with the meta-cognitive state into the global
        self-state vector (see SelfModificationController.collect_state).
        """
        hist = self.performance_history
        last = hist[-1] if hist else 0.0
        return {
            "competence": float(last),
            "uncertainty": float(np.clip(1.0 - last, 0.0, 1.0)),
            "task_diversity": float(self.task_diversity()),
            "log_accepted": float(np.log1p(len(self.accepted_embeddings))),
            "trust": float(self.trust),
            "activation_norm": float(self.activation_count / 200.0),
            "perf_trend": float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0,
            "queries_norm": float(self.queries_answered / max(1, self.activation_count)),
            "corrections_norm": float(self.corrections_received / max(1, self.activation_count)),
            "bookkeeping_norm": float(len(self.internalization_level) / 200.0),
        }


class WorldKnowledgeNetwork(CognitiveNetwork):
    """N1: general / world knowledge."""


class MathNetwork(CognitiveNetwork):
    """N2: mathematics."""


class LogicNetwork(CognitiveNetwork):
    """N3: logic & reasoning."""


class LanguageNetwork(CognitiveNetwork):
    """N4: language understanding."""


class VerificationNetwork(CognitiveNetwork):
    """N5: fact-checking (also hosts the confidence aggregation logic)."""
