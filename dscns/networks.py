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
