"""Checkpoint Manager (v0.6.0 / Phase 6).

Manages three types of checkpoints:
  1. Best Checkpoint: selected by validation score (no test leakage)
  2. Final Checkpoint: the state at the last round
  3. Relay Checkpoint: state for cross-version continuation

Design principles:
  - Best ≠ Final ≠ Relay (three different model states)
  - Best is selected by a fixed score formula on VALIDATION data
  - Final is always round 450 (or the last round)
  - Relay captures full training state for resumption
  - All checkpoints have SHA256 checksums
  - All checkpoints have metadata.json

Components:
  CheckpointManager       -- save/load/manage checkpoints
  CheckpointMetadata      -- metadata for a single checkpoint
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    version: str = "v0.6.0"
    phase: str = "P6"
    condition: str = ""
    seed: int = 0
    round: int = 0
    checkpoint_type: str = ""   # "best", "final", "relay"
    # scoring
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)
    # source tracking
    source_checkpoint: str = ""
    architecture_hash: str = ""
    git_commit: str = ""
    # timing
    timestamp: str = ""
    # integrity
    sha256: str = ""
    file_size: int = 0
    # experiment config
    config_hash: str = ""
    n_rounds_completed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "phase": self.phase,
            "condition": self.condition,
            "seed": self.seed,
            "round": self.round,
            "checkpoint_type": self.checkpoint_type,
            "score": self.score,
            "score_components": self.score_components,
            "source_checkpoint": self.source_checkpoint,
            "architecture_hash": self.architecture_hash,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "config_hash": self.config_hash,
            "n_rounds_completed": self.n_rounds_completed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointMetadata":
        meta = cls()
        for k, v in d.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        return meta


class CheckpointManager:
    """Manages Best/Final/Relay checkpoints.

    Directory structure:
        base_dir/
        ├── best/
        │   ├── model.pt
        │   └── metadata.json
        ├── final/
        │   ├── model.pt
        │   └── metadata.json
        └── relay/
            ├── model_state.pt
            ├── policy_state.pt
            ├── optimizer_state.pt
            ├── memory_snapshot.pt
            ├── experience_value.pt
            ├── round_counter.json
            ├── random_state.pt
            ├── config.yaml
            ├── architecture.json
            ├── metrics.json
            └── lineage.json
    """

    def __init__(self, base_dir: str, version: str = "v0.6.0"):
        self.base_dir = base_dir
        self.version = version
        self.best_dir = os.path.join(base_dir, "best")
        self.final_dir = os.path.join(base_dir, "final")
        self.relay_dir = os.path.join(base_dir, "relay")

        # best score tracking
        self.best_score: float = -float("inf")
        self.best_round: int = 0
        self.score_history: List[Dict[str, Any]] = []

    def compute_best_score(
        self,
        performance: float,
        rfr: float = 0.0,
        drift: float = 0.0,
        stability: float = 0.0,
        w_perf: float = 1.0,
        w_rfr: float = 0.5,
        w_drift: float = 0.1,
        w_stab: float = 0.3,
    ) -> float:
        """Compute Best Score for checkpoint selection.

        Score = w_p * Performance - w_r * RFR - w_d * Drift + w_s * Stability

        Must use VALIDATION data only (no test leakage).
        """
        score = (w_perf * performance -
                 w_rfr * rfr -
                 w_drift * drift +
                 w_stab * stability)
        return float(score)

    def should_update_best(self, current_score: float) -> bool:
        """Check if current score beats the best."""
        if current_score > self.best_score:
            self.best_score = current_score
            return True
        return False

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        metadata: CheckpointMetadata,
        checkpoint_type: str = "best",
        filename: str = "model.pt",
    ) -> str:
        """Save a checkpoint with metadata.

        Args:
            state_dict: the state to save (model weights, etc.)
            metadata: checkpoint metadata.
            checkpoint_type: "best", "final", or "relay".
            filename: name of the state file.

        Returns:
            Path to the saved checkpoint directory.
        """
        target_dir = {"best": self.best_dir, "final": self.final_dir,
                      "relay": self.relay_dir}[checkpoint_type]
        os.makedirs(target_dir, exist_ok=True)

        # save state
        state_path = os.path.join(target_dir, filename)
        if HAS_TORCH and isinstance(state_dict, dict):
            torch.save(state_dict, state_path)
        else:
            with open(state_path, "w") as f:
                json.dump(state_dict, f, indent=2, default=str)

        # compute SHA256
        sha256 = self._compute_sha256(state_path)
        file_size = os.path.getsize(state_path)

        # update metadata
        metadata.checkpoint_type = checkpoint_type
        metadata.sha256 = sha256
        metadata.file_size = file_size
        metadata.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.version = self.version

        # save metadata
        meta_path = os.path.join(target_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        return target_dir

    def save_best(
        self,
        state_dict: Dict[str, Any],
        condition: str,
        seed: int,
        round_id: int,
        score: float,
        score_components: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> str:
        """Save best checkpoint if score is the new best."""
        if not self.should_update_best(score):
            return ""

        meta = CheckpointMetadata(
            condition=condition,
            seed=seed,
            round=round_id,
            checkpoint_type="best",
            score=score,
            score_components=score_components or {},
            n_rounds_completed=round_id,
        )
        for k, v in kwargs.items():
            if hasattr(meta, k):
                setattr(meta, k, v)

        self.best_round = round_id
        self.score_history.append({
            "round": round_id, "score": score,
            "components": score_components or {},
        })

        return self.save_checkpoint(state_dict, meta, "best")

    def save_final(
        self,
        state_dict: Dict[str, Any],
        condition: str,
        seed: int,
        round_id: int,
        score: float = 0.0,
        **kwargs,
    ) -> str:
        """Save final checkpoint (always saved, may overwrite)."""
        meta = CheckpointMetadata(
            condition=condition,
            seed=seed,
            round=round_id,
            checkpoint_type="final",
            score=score,
            n_rounds_completed=round_id,
        )
        for k, v in kwargs.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        return self.save_checkpoint(state_dict, meta, "final")

    def save_relay(
        self,
        relay_state: Dict[str, Any],
        condition: str,
        seed: int,
        round_id: int,
        lineage: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Save relay checkpoint (full state for continuation).

        Relay state must include:
            model_state, policy_state, optimizer_state,
            memory_snapshot, experience_value,
            round_counter, random_state, config, metrics
        """
        os.makedirs(self.relay_dir, exist_ok=True)

        # save individual components
        for key in ["model_state", "policy_state", "optimizer_state",
                     "memory_snapshot", "experience_value"]:
            if key in relay_state:
                path = os.path.join(self.relay_dir, f"{key}.pt")
                if HAS_TORCH:
                    torch.save(relay_state[key], path)
                else:
                    with open(path, "w") as f:
                        json.dump(relay_state[key], f, default=str)

        # save round counter
        if "round_counter" in relay_state:
            path = os.path.join(self.relay_dir, "round_counter.json")
            with open(path, "w") as f:
                json.dump(relay_state["round_counter"], f)

        # save random state
        if "random_state" in relay_state:
            path = os.path.join(self.relay_dir, "random_state.pt")
            if HAS_TORCH:
                torch.save(relay_state["random_state"], path)

        # save config
        if "config" in relay_state:
            path = os.path.join(self.relay_dir, "config.yaml")
            with open(path, "w") as f:
                if isinstance(relay_state["config"], str):
                    f.write(relay_state["config"])
                else:
                    json.dump(relay_state["config"], f, indent=2, default=str)

        # save architecture
        if "architecture" in relay_state:
            path = os.path.join(self.relay_dir, "architecture.json")
            with open(path, "w") as f:
                json.dump(relay_state["architecture"], f, indent=2, default=str)

        # save metrics
        if "metrics" in relay_state:
            path = os.path.join(self.relay_dir, "metrics.json")
            with open(path, "w") as f:
                json.dump(relay_state["metrics"], f, indent=2, default=str)

        # save lineage
        if lineage is None:
            lineage = {}
        lineage["version"] = self.version
        lineage["round"] = round_id
        lineage["condition"] = condition
        lineage["seed"] = seed
        path = os.path.join(self.relay_dir, "lineage.json")
        with open(path, "w") as f:
            json.dump(lineage, f, indent=2, default=str)

        # save metadata
        meta = CheckpointMetadata(
            condition=condition, seed=seed, round=round_id,
            checkpoint_type="relay", n_rounds_completed=round_id,
        )
        for k, v in kwargs.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        meta_path = os.path.join(self.relay_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2, default=str)

        return self.relay_dir

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint state."""
        return self._load_state(self.best_dir)

    def load_final(self) -> Optional[Dict[str, Any]]:
        """Load final checkpoint state."""
        return self._load_state(self.final_dir)

    def load_relay(self) -> Optional[Dict[str, Any]]:
        """Load relay checkpoint."""
        result = {}
        for key in ["model_state", "policy_state", "optimizer_state",
                     "memory_snapshot", "experience_value"]:
            path = os.path.join(self.relay_dir, f"{key}.pt")
            if os.path.exists(path):
                if HAS_TORCH:
                    result[key] = torch.load(path, map_location="cpu")
                else:
                    with open(path) as f:
                        result[key] = json.load(f)

        for key in ["round_counter", "random_state"]:
            path = os.path.join(self.relay_dir, f"{key}.pt")
            if os.path.exists(path) and HAS_TORCH:
                result[key] = torch.load(path, map_location="cpu")

        for key in ["config", "architecture", "metrics", "lineage"]:
            ext = "json" if key != "config" else "yaml"
            path = os.path.join(self.relay_dir, f"{key}.{ext}")
            if not os.path.exists(path):
                path = os.path.join(self.relay_dir, f"{key}.json")
            if os.path.exists(path):
                with open(path) as f:
                    try:
                        result[key] = json.load(f)
                    except json.JSONDecodeError:
                        result[key] = f.read()

        return result if result else None

    def load_metadata(self, checkpoint_type: str = "best") -> Optional[CheckpointMetadata]:
        """Load metadata for a checkpoint type."""
        dir_map = {"best": self.best_dir, "final": self.final_dir,
                   "relay": self.relay_dir}
        target_dir = dir_map.get(checkpoint_type, self.best_dir)
        meta_path = os.path.join(target_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            return CheckpointMetadata.from_dict(json.load(f))

    def verify_integrity(self, checkpoint_type: str = "best") -> bool:
        """Verify checkpoint integrity via SHA256."""
        meta = self.load_metadata(checkpoint_type)
        if meta is None:
            return False
        dir_map = {"best": self.best_dir, "final": self.final_dir,
                   "relay": self.relay_dir}
        target_dir = dir_map[checkpoint_type]
        state_path = os.path.join(target_dir, "model.pt")
        if not os.path.exists(state_path):
            return False
        actual_sha = self._compute_sha256(state_path)
        return actual_sha == meta.sha256

    def artifact_manifest(self) -> Dict[str, Any]:
        """Generate artifact manifest for release."""
        manifest = {
            "version": self.version,
            "artifacts": [],
        }
        for ctype in ["best", "final", "relay"]:
            meta = self.load_metadata(ctype)
            if meta:
                manifest["artifacts"].append({
                    "type": ctype,
                    "metadata": meta.to_dict(),
                })
        manifest["score_history"] = self.score_history
        return manifest

    def _load_state(self, directory: str) -> Optional[Dict[str, Any]]:
        """Load state from a directory."""
        if not os.path.exists(directory):
            return None
        state_path = os.path.join(directory, "model.pt")
        if not os.path.exists(state_path):
            return None
        if HAS_TORCH:
            return torch.load(state_path, map_location="cpu")
        with open(state_path) as f:
            return json.load(f)

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def summary(self) -> Dict[str, Any]:
        """Summary of checkpoint state."""
        return {
            "version": self.version,
            "base_dir": self.base_dir,
            "best_score": self.best_score,
            "best_round": self.best_round,
            "n_score_updates": len(self.score_history),
            "has_best": os.path.exists(os.path.join(self.best_dir, "model.pt")),
            "has_final": os.path.exists(os.path.join(self.final_dir, "model.pt")),
            "has_relay": os.path.exists(os.path.join(self.relay_dir, "metadata.json")),
        }
