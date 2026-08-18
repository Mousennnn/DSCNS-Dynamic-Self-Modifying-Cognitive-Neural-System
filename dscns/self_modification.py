"""Learned model-driven self-modification (Phase 4).

Implements the design-report modification proposal:

    self-state -> SelfStateEncoder -> z_self -> SelfModificationPolicy
    -> ArchitectureAction -> safety constraints -> candidate architecture
    -> short adaptation -> regression evaluation -> accept / rollback
    -> reward -> policy update.

The decision power over structural modifications is moved from the rule
engine to a small trainable neural policy:

  * SelfStateEncoder  -- maps raw system statistics to a fixed-size
                         internal self-representation z_self;
  * SelfModificationPolicy -- outputs P(action), P(target),
                         P(secondary target), P(domain), magnitude and a
                         learned value baseline;
  * SelfModificationController -- orchestrates state collection, action
                         proposal, rule imitation (Stage A), REINFORCE
                         updates with the modification reward (Stage B),
                         and the modification trace.

Reward (proposal section 11):

    Reward = dPerf - l1*forgetting - l2*parameter_growth
                   - l3*compute_cost - l4*instability

No prompts, no generated code, no exec(): the policy produces numbers by
forward propagation only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modification_memory import ModificationMemory, ModificationRecord

# ---------------------------------------------------------------------- #
# action space (proposal section 3.2)
# ---------------------------------------------------------------------- #
ACTIONS = ["no_op", "expand", "contract", "split", "merge",
           "connect", "disconnect"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# rough normalized compute cost of each operation (reward term l3)
OP_COST = {"no_op": 0.0, "connect": 0.05, "disconnect": 0.02,
           "expand": 0.15, "contract": 0.05, "split": 0.20, "merge": 0.10}

# fixed state dimensions (see SelfModificationController.collect_state)
FEAT_DIM = 12   # per-network feature vector
DOM_DIM = 5     # per-domain feature vector
STATE_DIM = 47  # global self-state vector


@dataclass
class ArchitectureAction:
    """Stable interface between the policy and the structure executor."""

    operation: str = "no_op"
    target: Optional[str] = None
    secondary_target: Optional[str] = None
    magnitude: float = 0.5
    confidence: float = 0.5
    source: str = "rule"            # rule | policy
    round: int = -1
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _sample_categorical(probs: np.ndarray, rng: np.random.RandomState) -> int:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        return int(rng.randint(len(probs)))
    return int(rng.choice(len(probs), p=probs / s))


def _pick_label(probs: np.ndarray, labels: Sequence[str], greedy: bool,
                rng: np.random.RandomState) -> Optional[str]:
    labels = list(labels)
    if not labels:
        return None
    if greedy:
        return labels[int(np.argmax(probs))]
    return labels[_sample_categorical(probs, rng)]


# ---------------------------------------------------------------------- #
# neural self-modification policy
# ---------------------------------------------------------------------- #
class SelfStateEncoder(nn.Module):
    """Linear -> LayerNorm -> MLP -> z_self (proposal section 5)."""

    def __init__(self, state_dim: int, hidden: int = 64, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class SelfModificationPolicy(nn.Module):
    """P(action), P(target), magnitude and value heads over z_self.

    Target selection uses a small attention-style scoring of the current
    per-network feature matrix F_net (pointer-network style) so that the
    policy works for a varying number of networks.
    """

    def __init__(self, state_dim: int = STATE_DIM, feat_dim: int = FEAT_DIM,
                 dom_dim: int = DOM_DIM, hidden: int = 64, z_dim: int = 32,
                 temperature: float = 0.8):
        super().__init__()
        self.encoder = SelfStateEncoder(state_dim, hidden, z_dim)
        self.temperature = temperature
        self.action_head = nn.Linear(z_dim, len(ACTIONS))
        self.magnitude_head = nn.Sequential(
            nn.Linear(z_dim, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.value_head = nn.Linear(z_dim, 1)
        # per-network target scoring: score_i = F_i @ w + z @ wz
        self.target_w = nn.Linear(feat_dim, 1, bias=False)
        self.target_z = nn.Linear(z_dim, 1, bias=False)
        self.sec_w = nn.Linear(feat_dim, 1, bias=False)
        self.sec_z = nn.Linear(z_dim, 1, bias=False)
        # per-domain scoring (EXPAND)
        self.dom_w = nn.Linear(dom_dim, 1, bias=False)
        self.dom_z = nn.Linear(z_dim, 1, bias=False)

    # ------------------------------------------------------------------ #
    def forward(self, s: torch.Tensor, F_net: Optional[torch.Tensor] = None,
                F_dom: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        z = self.encoder(s)
        out: Dict[str, Any] = {
            "z": z,
            "action_logits": self.action_head(z) / self.temperature,
            "magnitude": self.magnitude_head(z).squeeze(-1),
            "value": self.value_head(z).squeeze(-1),
        }
        if F_net is not None and F_net.shape[0] > 0:
            z_n = self.target_z(z).squeeze(-1).expand(F_net.shape[0])
            out["target_logits"] = self.target_w(F_net).squeeze(-1) + z_n
            z_s = self.sec_z(z).squeeze(-1).expand(F_net.shape[0])
            out["sec_logits"] = self.sec_w(F_net).squeeze(-1) + z_s
        if F_dom is not None and F_dom.shape[0] > 0:
            z_d = self.dom_z(z).squeeze(-1).expand(F_dom.shape[0])
            out["dom_logits"] = self.dom_w(F_dom).squeeze(-1) + z_d
        return out

    # ------------------------------------------------------------------ #
    def act(self, s: np.ndarray, F_net: Optional[np.ndarray],
            net_ids: Sequence[str], F_dom: Optional[np.ndarray],
            dom_ids: Sequence[str], greedy: bool = False,
            rng: Optional[np.random.RandomState] = None) -> ArchitectureAction:
        """Sample one ArchitectureAction by forward propagation (no rules)."""
        rng = rng or np.random.RandomState(0)
        s_t = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0)
        F_t = (torch.from_numpy(np.asarray(F_net, dtype=np.float32))
               if F_net is not None and len(F_net) else None)
        Fd_t = (torch.from_numpy(np.asarray(F_dom, dtype=np.float32))
                if F_dom is not None and len(F_dom) else None)
        with torch.no_grad():
            out = self.forward(s_t, F_t, Fd_t)
            probs = torch.softmax(out["action_logits"], dim=-1)[0].cpu().numpy()
            op_idx = (int(np.argmax(probs)) if greedy
                      else _sample_categorical(probs, rng))
            op = ACTIONS[op_idx]
            target = sec = None
            if op in ("split", "contract") and out.get("target_logits") is not None:
                p = torch.softmax(out["target_logits"], dim=-1).cpu().numpy()
                target = _pick_label(p, net_ids, greedy, rng)
            elif op in ("merge", "connect", "disconnect") and \
                    out.get("target_logits") is not None:
                p = torch.softmax(out["target_logits"], dim=-1).cpu().numpy()
                target = _pick_label(p, net_ids, greedy, rng)
                p2 = torch.softmax(out["sec_logits"], dim=-1).cpu().numpy()
                sec = _pick_label(p2, net_ids, greedy, rng)
            elif op == "expand" and out.get("dom_logits") is not None:
                p = torch.softmax(out["dom_logits"], dim=-1).cpu().numpy()
                target = _pick_label(p, dom_ids, greedy, rng)
            mag = float(np.clip(out["magnitude"].item(), 0.0, 1.0))
            conf = float(probs[op_idx])
        return ArchitectureAction(
            operation=op, target=target, secondary_target=sec,
            magnitude=mag, confidence=conf, source="policy",
        )

    def action_entropy(self, s: np.ndarray) -> float:
        s_t = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(s_t)
            probs = torch.softmax(out["action_logits"], dim=-1)
            ent = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
        return float(ent.item())


# ---------------------------------------------------------------------- #
# controller
# ---------------------------------------------------------------------- #
class SelfModificationController:
    """Orchestrates state collection, action proposal and policy learning."""

    def __init__(self, domains: Sequence[str], config: Any, seed: int = 42):
        self.domains = list(domains)
        self.config = config
        self.rng = np.random.RandomState(seed)

        self.policy = SelfModificationPolicy(
            state_dim=STATE_DIM, feat_dim=FEAT_DIM, dom_dim=DOM_DIM,
            hidden=getattr(config, "policy_hidden", 64),
            temperature=getattr(config, "policy_temperature", 0.8),
        )
        torch.manual_seed(seed)
        self._opt = torch.optim.AdamW(
            self.policy.parameters(), lr=getattr(config, "policy_lr", 3e-4),
            weight_decay=1e-4,
        )
        self.policy.eval()

        self.memory = ModificationMemory()

        # reward bookkeeping
        self._lambda_forgetting = getattr(config, "reward_lambda_forgetting", 0.5)
        self._lambda_params = getattr(config, "reward_lambda_params", 0.3)
        self._lambda_compute = getattr(config, "reward_lambda_compute", 0.1)
        self._lambda_instability = getattr(config, "reward_lambda_instability", 0.3)
        self._adaptation_window = getattr(config, "adaptation_window", 3)

        # per-round bookkeeping
        self._probe_series: List[float] = []
        self._perf_series: List[Dict[str, float]] = []
        self._perf_latest: Dict[str, float] = {}
        self._recent_deltas: List[float] = []      # per-round probe deltas
        self._imitation_buffer: List[Tuple[Any, ...]] = []
        self.trace: List[Dict[str, Any]] = []
        self._rl_updates = 0
        self._round = -1
        self._last_state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # state collection (proposal section 5)
    # ------------------------------------------------------------------ #
    def track_perf(self, perf_by_domain: Dict[str, float]) -> None:
        """Register this round's per-domain performance (once per round)."""
        self._perf_series.append(dict(perf_by_domain))
        self._perf_latest = dict(perf_by_domain)

    def collect_state(self, system: Any):
        """Return (s, F_net, net_ids, F_dom, dom_ids)."""
        perf_by_domain = self._perf_latest
        nets = system.networks
        meta = system.meta_state

        net_ids = sorted(nets.keys())
        F_net = [self._net_features(net, meta, nets) for net_id in net_ids
                 for net in [nets[net_id]]]
        dom_ids = self.domains
        F_dom = [self._dom_features(d, system, meta, perf_by_domain)
                 for d in dom_ids]

        s = self._global_state(system, meta, net_ids, nets, perf_by_domain)
        assert len(s) == STATE_DIM, f"state dim mismatch: {len(s)} != {STATE_DIM}"
        return (np.asarray(s, dtype=np.float32),
                np.asarray(F_net, dtype=np.float32) if F_net
                else np.zeros((0, FEAT_DIM), dtype=np.float32),
                net_ids,
                np.asarray(F_dom, dtype=np.float32) if F_dom
                else np.zeros((0, DOM_DIM), dtype=np.float32),
                dom_ids)

    # ------------------------------------------------------------------ #
    def _net_features(self, net: Any, meta: Any,
                      nets: Dict[str, Any]) -> List[float]:
        hist = net.performance_history
        comp = meta.network_competence.get(net.id, hist[-1] if hist else 0.0)
        unc = meta.network_uncertainty.get(net.id, 0.5)
        div = net.task_diversity()
        trend = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        red = self._pairwise_sim_contribution(net, nets)
        return [
            float(comp), float(unc), float(div),
            float(np.log1p(len(net.accepted_embeddings)) / 5.0),
            float(net.trust),
            float(net.activation_count / max(200, len(nets) * 40)),
            float(np.clip(trend * 10.0, -1.0, 1.0)),
            float(1.0 - div),
            float(red),
            float(net.queries_answered / max(1, net.activation_count)),
            float(net.corrections_received / max(1, net.activation_count)),
            float(len(net.internalization_level) / 200.0),
        ]

    @staticmethod
    def _pairwise_sim_contribution(net: Any, nets: Dict[str, Any]) -> float:
        from .utils import cosine_similarity

        e = net.representation_embedding()
        if e.size == 1:
            return 0.0
        sims = []
        for nid, other in nets.items():
            if nid == net.id:
                continue
            oe = other.representation_embedding()
            if oe.size > 1:
                sims.append(cosine_similarity(e, oe))
        return float(np.mean(sims)) if sims else 0.0

    def _dom_features(self, d: str, system: Any, meta: Any,
                      perf_by_domain: Dict[str, float]) -> List[float]:
        cov = meta.knowledge_coverage.get(d, 0.0)
        perf = perf_by_domain.get(d, 0.0)
        forget = self._domain_forgetting(d)
        episodes = []
        if system.networks:
            first = next(iter(system.networks.values()))
            episodes = first.memory.episodic.episodes
        n_ep = sum(1 for e in episodes if e.get("context") == d)
        return [float(cov), float(perf), float(forget),
                float(1.0 - cov), float(n_ep / 200.0)]

    # ------------------------------------------------------------------ #
    def _global_state(self, system: Any, meta: Any, net_ids: List[str],
                      nets: Dict[str, Any],
                      perf_by_domain: Dict[str, float]) -> List[float]:
        budget = float(getattr(self.config, "modification_budget_max", 8))
        f: List[float] = []

        def stats(xs: List[float]) -> List[float]:
            if not xs:
                return [0.0, 0.0, 0.0]
            return [float(np.mean(xs)), float(np.max(xs)), float(np.std(xs))]

        comps = [meta.network_competence.get(n, 0.0) for n in net_ids]
        uncs = [meta.network_uncertainty.get(n, 0.5) for n in net_ids]
        divs = [nets[n].task_diversity() for n in net_ids]
        nacc = [float(np.log1p(len(nets[n].accepted_embeddings))) for n in net_ids]
        trus = [nets[n].trust for n in net_ids]
        f += stats(comps) + stats(uncs) + stats(divs) + stats(nacc) + stats(trus)

        n_net = len(net_ids)
        f.append(n_net / max(1.0, budget))
        max_pairs = max(1.0, n_net * (n_net - 1) / 2.0)
        f.append(len(system.connections) / max_pairs)
        w = list(system.connections.values())
        f.append(float(np.mean(w)) if w else 0.0)

        lp = meta.learning_progress
        f.append(float(np.mean(np.diff(lp[-10:]))) if len(lp) >= 2 else 0.0)
        f.append(self._probe_series[-1] if self._probe_series else 0.0)
        f.append(self._mean_forgetting())

        covs = [meta.knowledge_coverage.get(d, 0.0) for d in self.domains]
        f.append(float(np.mean(covs)) if covs else 0.0)
        f.append(float(min(covs)) if covs else 0.0)

        pair_sims = self._all_pairwise_sims(nets)
        f.append(float(np.mean([1.0 - x for x in pair_sims])) if pair_sims else 0.0)
        f.append(float(np.mean(pair_sims)) if pair_sims else 0.0)

        from .evaluation import structural_metrics
        sm = structural_metrics(nets, system.domain_embeddings)
        f.append(float(sm["mean_specialization"]))

        total_rounds = max(1, getattr(self.config, "total_rounds", 16))
        f.append(self._round / max(1.0, total_rounds))
        since = self._round - getattr(system, "_last_evolve_change", -10)
        f.append(float(np.clip(since / 8.0, 0.0, 1.0)))
        f.append(self._param_utilization(system))
        f.append(self.memory.success_rate(window=8))
        f.append(self.memory.mean_reward(window=8))
        f.append(1.0 if getattr(system, "_pending_mod", None) is not None else 0.0)

        for d in self.domains:
            f.append(meta.knowledge_coverage.get(d, 0.0))
        for d in self.domains:
            f.append(perf_by_domain.get(d, 0.0))
        for d in self.domains:
            f.append(self._domain_forgetting(d))
        return f

    # ------------------------------------------------------------------ #
    @staticmethod
    def _all_pairwise_sims(nets: Dict[str, Any]) -> List[float]:
        from .utils import cosine_similarity

        ids = list(nets.keys())
        sims = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = nets[ids[i]], nets[ids[j]]
                ea, eb = a.representation_embedding(), b.representation_embedding()
                if ea.size > 1 and eb.size > 1:
                    sims.append(cosine_similarity(ea, eb))
        return sims

    @staticmethod
    def _param_utilization(system: Any) -> float:
        n = 0
        for name, p in system.base_model.peft_model.named_parameters():
            if "lora" in name:
                n += p.numel()
        per_adapter = max(1, n // max(1, len(system.networks) + 2))
        budget = getattr(system.config, "modification_budget_max", 8)
        return float(np.clip(n / (per_adapter * budget), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # forgetting / probe bookkeeping
    # ------------------------------------------------------------------ #
    def track_probe(self, probe: float, pending: bool = False) -> None:
        self._probe_series.append(float(probe))
        if len(self._probe_series) >= 2 and not pending:
            self._recent_deltas.append(self._probe_series[-1] - self._probe_series[-2])
            self._recent_deltas = self._recent_deltas[-6:]

    def baseline_delta(self) -> float:
        """Mean per-round probe delta of recent non-pending rounds."""
        if not self._recent_deltas:
            return 0.0
        return float(np.mean(self._recent_deltas))

    def current_forgetting(self) -> float:
        return self._mean_forgetting()

    def _mean_forgetting(self) -> float:
        if len(self._perf_series) < 2:
            return 0.0
        fs = []
        for d in self.domains:
            col = [row.get(d, 0.0) for row in self._perf_series]
            fs.append(max(col) - col[-1])
        return float(np.mean(fs)) if fs else 0.0

    def _domain_forgetting(self, d: str) -> float:
        if len(self._perf_series) < 2:
            return 0.0
        col = [row.get(d, 0.0) for row in self._perf_series]
        return float(max(col) - col[-1])

    def probe_window(self, r_from: int, r_to: int) -> List[float]:
        return [self._probe_series[i] for i in range(r_from, min(r_to, len(self._probe_series)) + 1)]

    # ------------------------------------------------------------------ #
    # reward (proposal section 11)
    # ------------------------------------------------------------------ #
    def compute_reward(self, op: str, probe_before: float, probe_after: float,
                       forgetting_before: float, forgetting_after: float,
                       params_before: float, params_after: float,
                       window_probes: List[float],
                       window_rounds: int) -> Tuple[float, Dict[str, float]]:
        delta = probe_after - probe_before
        baseline = self.baseline_delta() * max(1, window_rounds)
        marginal = delta - baseline
        fg = max(0.0, forgetting_after - forgetting_before)
        pg = max(0.0, (params_after - params_before) / max(1e-9, params_before))
        cost = OP_COST.get(op, 0.0)
        inst = float(np.std(window_probes)) if len(window_probes) > 1 else 0.0
        r = (marginal
             - self._lambda_forgetting * fg
             - self._lambda_params * pg
             - self._lambda_compute * cost
             - self._lambda_instability * inst)
        return float(r), {
            "delta": float(delta), "marginal": float(marginal),
            "forgetting_change": float(fg), "param_growth": float(pg),
            "compute_cost": float(cost), "instability": float(inst),
        }

    # ------------------------------------------------------------------ #
    # rule decision (single-action protocol, priority split > merge > connect)
    # ------------------------------------------------------------------ #
    def rule_decision(self, evolver: Any, system: Any,
                      perf_by_domain: Dict[str, float]) -> ArchitectureAction:
        nets = system.networks
        # 1. split (at most one)
        for nid in list(nets.keys()):
            net = nets[nid]
            if evolver.should_split(net, perf_by_domain, system.round_idx):
                return ArchitectureAction(
                    "split", target=nid, magnitude=0.5, confidence=1.0,
                    source="rule", round=system.round_idx)
        # 2. merge (at most one)
        ids = list(nets.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                co = system.bus.get_co_activation_matrix().get((a, b), 0)
                if evolver.should_merge(nets[a], nets[b], co):
                    return ArchitectureAction(
                        "merge", target=a, secondary_target=b, magnitude=0.5,
                        confidence=1.0, source="rule", round=system.round_idx)
        # 3. connect the strongest eligible pair
        pair, w = self._strongest_connection(evolver, nets, system)
        if pair is not None:
            return ArchitectureAction(
                "connect", target=pair[0], secondary_target=pair[1],
                magnitude=w, confidence=1.0, source="rule",
                round=system.round_idx)
        return ArchitectureAction("no_op", source="rule",
                                  round=system.round_idx)

    @staticmethod
    def _strongest_connection(evolver: Any, nets: Dict[str, Any],
                              system: Any):
        ids = list(nets.keys())
        co = system.bus.get_co_activation_matrix()
        best, best_w = None, None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                c = float(co.get((a, b), 0) + co.get((b, a), 0))
                flow = 0.0
                w = evolver.co_act_w * c + evolver.info_flow_w * flow
                if w >= evolver.connect_threshold:
                    if best_w is None or w > best_w:
                        best, best_w = (a, b), w
        return best, best_w

    # ------------------------------------------------------------------ #
    # Stage A: imitation learning (proposal section 8)
    # ------------------------------------------------------------------ #
    def record_imitation(self, state_pack: Tuple[Any, ...],
                         action: ArchitectureAction) -> None:
        self._imitation_buffer.append(state_pack + (action,))
        if len(self._imitation_buffer) > 96:
            self._imitation_buffer = self._imitation_buffer[-96:]

    def train_imitation(self) -> float:
        buf = self._imitation_buffer
        if len(buf) < 2:
            return 0.0
        self.policy.train()
        total, n = 0.0, 0
        for entry in buf:
            s, F_net, net_ids, F_dom, dom_ids, action = entry
            s_t = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0)
            F_t = (torch.from_numpy(np.asarray(F_net, dtype=np.float32))
                   if F_net.shape[0] else None)
            Fd_t = (torch.from_numpy(np.asarray(F_dom, dtype=np.float32))
                    if F_dom.shape[0] else None)
            out = self.policy.forward(s_t, F_t, Fd_t)
            loss = F.cross_entropy(
                out["action_logits"],
                torch.tensor([ACTION_INDEX[action.operation]], dtype=torch.long))
            net_ids = list(net_ids)
            if action.operation in ("split", "contract") and \
                    out.get("target_logits") is not None and \
                    action.target in net_ids:
                ti = net_ids.index(action.target)
                loss = loss + F.cross_entropy(
                    out["target_logits"].unsqueeze(0),
                    torch.tensor([ti], dtype=torch.long))
            elif action.operation in ("merge", "connect", "disconnect") and \
                    out.get("target_logits") is not None and \
                    action.target in net_ids and \
                    action.secondary_target in net_ids:
                ti = net_ids.index(action.target)
                si = net_ids.index(action.secondary_target)
                loss = loss + F.cross_entropy(
                    out["target_logits"].unsqueeze(0),
                    torch.tensor([ti], dtype=torch.long))
                loss = loss + F.cross_entropy(
                    out["sec_logits"].unsqueeze(0),
                    torch.tensor([si], dtype=torch.long))
            if action.operation in ("split", "merge", "connect", "expand"):
                mag = torch.tensor([float(np.clip(action.magnitude, 0.0, 1.0))])
                loss = loss + 0.1 * F.mse_loss(out["magnitude"], mag)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            total += float(loss.item())
            n += 1
        self.policy.eval()
        return total / max(1, n)

    # ------------------------------------------------------------------ #
    # Stage B: REINFORCE (proposal sections 9 / 12)
    # ------------------------------------------------------------------ #
    def propose(self, system: Any, state_pack: Tuple[Any, ...]) -> ArchitectureAction:
        """Stage B action proposal: policy sampling + epsilon exploration."""
        s, F_net, net_ids, F_dom, dom_ids = state_pack
        eps = getattr(self.config, "policy_epsilon", 0.15)
        if self.rng.rand() < eps:
            # uniform exploration over the architecture action space
            op = ACTIONS[int(self.rng.randint(len(ACTIONS)))]
            action = ArchitectureAction(op, source="policy",
                                        round=system.round_idx)
            if op in ("split", "contract") and net_ids:
                action.target = net_ids[int(self.rng.randint(len(net_ids)))]
            elif op in ("merge", "connect", "disconnect") and len(net_ids) >= 2:
                i, j = self.rng.choice(len(net_ids), size=2, replace=False)
                action.target, action.secondary_target = net_ids[int(i)], net_ids[int(j)]
            elif op == "expand" and dom_ids:
                action.target = dom_ids[int(self.rng.randint(len(dom_ids)))]
            action.magnitude = float(self.rng.uniform(0.3, 0.8))
            action.confidence = 1.0 / len(ACTIONS)
            return action
        action = self.policy.act(s, F_net, net_ids, F_dom, dom_ids,
                                 greedy=False, rng=self.rng)
        action.source = "policy"
        action.round = system.round_idx
        return action

    def record_policy_noop(self, state: np.ndarray, reason: str = "") -> None:
        self.memory.add(ModificationRecord(
            round=self._round, op="no_op", source="policy", state=state,
            accepted=True, reward=0.0, reason=reason or "noop"))

    def record_modification(self, action: ArchitectureAction, accepted: bool,
                            reward: float, comps: Dict[str, float],
                            probe_before: float, probe_after: float,
                            state: Optional[np.ndarray] = None) -> None:
        self.memory.add(ModificationRecord(
            round=action.round, op=action.operation, target=action.target,
            secondary_target=action.secondary_target, magnitude=action.magnitude,
            confidence=action.confidence, source=action.source, state=state,
            accepted=accepted, reward=reward,
            probe_before=probe_before, probe_after=probe_after,
            delta_perf=comps.get("delta"), forgetting_change=comps.get("forgetting_change"),
            param_growth=comps.get("param_growth"), compute_cost=comps.get("compute_cost"),
            instability=comps.get("instability"), reason=action.reason))

    def train_rl(self) -> float:
        """REINFORCE update with a learned value baseline over recent samples."""
        samples = self.memory.rl_samples()[-16:]
        if not samples:
            return 0.0
        self.policy.train()
        total, n = 0.0, 0
        for _ in range(2):  # two epochs over the recent buffer
            for rec in samples:
                if rec.state is None:
                    continue
                s_t = torch.from_numpy(
                    np.asarray(rec.state, dtype=np.float32)).unsqueeze(0)
                r_t = torch.tensor([float(rec.reward)], dtype=torch.float32)
                out = self.policy.forward(s_t)
                probs = torch.softmax(out["action_logits"], dim=-1)
                log_prob = torch.log(probs[0, ACTION_INDEX[rec.op]] + 1e-8)
                value = out["value"]
                advantage = (r_t - value.detach())
                entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
                loss = (-advantage * log_prob
                        + 0.5 * F.mse_loss(value, r_t)
                        - 0.01 * entropy)
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
                total += float(loss.item())
                n += 1
        self.policy.eval()
        self._rl_updates += 1
        return total / max(1, n)

    # ------------------------------------------------------------------ #
    # trace
    # ------------------------------------------------------------------ #
    def record_round(self, round_idx: int, action: ArchitectureAction,
                     probe: float, n_networks: int, n_connections: int,
                     imitation_loss: float = 0.0, rl_loss: float = 0.0,
                     note: str = "") -> None:
        self._round = round_idx
        entry: Dict[str, Any] = {
            "round": round_idx, "op": action.operation,
            "target": action.target, "secondary_target": action.secondary_target,
            "magnitude": float(action.magnitude), "confidence": float(action.confidence),
            "source": action.source, "reason": action.reason,
            "probe": float(probe), "n_networks": int(n_networks),
            "n_connections": int(n_connections),
            "imitation_loss": float(imitation_loss), "rl_loss": float(rl_loss),
            "note": note,
        }
        if getattr(self, "_last_state", None) is not None:
            entry["policy_entropy"] = self.policy.action_entropy(self._last_state)
        self.trace.append(entry)
        if len(self.trace) > 400:
            self.trace = self.trace[-400:]

    def set_last_state(self, s: np.ndarray) -> None:
        self._last_state = np.asarray(s, dtype=np.float32)
