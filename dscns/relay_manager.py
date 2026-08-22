"""Relay Manager (v0.6.0 / Phase 6).

Implements the cross-version relay learning system:

    Relay_0 → 450 rounds → Relay_1 → 450 rounds → Relay_2 → ...

Each relay checkpoint captures the COMPLETE training state so that the next
version can CONTINUE from where the previous version left off.

This enables studying:
    "Can a DSCNS that has been self-modifying for a long time
     continue to learn effectively?"

Rules:
  1. Relay checkpoints are NEVER overwritten
  2. Each relay stores full lineage metadata
  3. Standard experiments start from base (fresh init)
  4. Relay experiments start from previous relay (continued)
  5. Both are reported separately (never mixed)

Components:
  RelayManager           -- manages relay checkpoints
  RelayLineage           -- tracks lineage across versions
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .checkpoint_manager import CheckpointManager


@dataclass
class RelayLineage:
    """Tracks the lineage of a relay checkpoint."""
    source_version: str = ""
    source_relay: str = ""
    target_version: str = ""
    continued_rounds: int = 0
    total_lineage_rounds: int = 0
    source_condition: str = ""
    source_seed: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_version": self.source_version,
            "source_relay": self.source_relay,
            "target_version": self.target_version,
            "continued_rounds": self.continued_rounds,
            "total_lineage_rounds": self.total_lineage_rounds,
            "source_condition": self.source_condition,
            "source_seed": self.source_seed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelayLineage":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


class RelayManager:
    """Manages cross-version relay checkpoints.

    Directory structure:
        relay_base/
        ├── relay_v0.6.0/
        │   ├── model_state.pt
        │   ├── policy_state.pt
        │   ├── ...
        │   └── lineage.json
        ├── relay_v0.6.0_stage_450/
        ├── relay_v0.6.0_stage_900/
        ├── relay_v0.6.0_stage_1350/
        └── relay_v0.6.0_stage_1800/

    Each stage saves a snapshot at the given round count.
    """

    def __init__(self, base_dir: str, version: str = "v0.6.0"):
        self.base_dir = base_dir
        self.version = version
        self.lineage = RelayLineage(target_version=version)
        self.stage_dirs: Dict[int, str] = {}

    def save_relay(
        self,
        relay_state: Dict[str, Any],
        condition: str,
        seed: int,
        round_id: int,
        stage_rounds: Optional[List[int]] = None,
    ) -> str:
        """Save relay checkpoint for the current version.

        Args:
            relay_state: full training state to save.
            condition: experimental condition.
            seed: random seed.
            round_id: total rounds completed.
            stage_rounds: list of round milestones to save stages at.

        Returns:
            Path to the saved relay directory.
        """
        if stage_rounds is None:
            stage_rounds = [450, 900, 1350, 1800]

        # determine stage name
        stage_name = f"relay_{self.version}"
        if round_id in stage_rounds:
            stage_name = f"relay_{self.version}_stage_{round_id}"

        relay_dir = os.path.join(self.base_dir, stage_name)
        os.makedirs(relay_dir, exist_ok=True)

        # save components - use CheckpointManager with relay_dir set directly
        ckpt_mgr = CheckpointManager(os.path.dirname(relay_dir), version=self.version)
        ckpt_mgr.relay_dir = relay_dir
        lineage = self.lineage.to_dict()
        lineage["continued_rounds"] = round_id
        lineage["source_condition"] = condition
        lineage["source_seed"] = seed
        lineage["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # save as relay type
        ckpt_mgr.save_relay(
            relay_state=relay_state,
            condition=condition,
            seed=seed,
            round_id=round_id,
            lineage=lineage,
        )

        self.stage_dirs[round_id] = relay_dir
        return relay_dir

    def load_latest_relay(self) -> Optional[Dict[str, Any]]:
        """Load the most recent relay checkpoint."""
        if not os.path.exists(self.base_dir):
            return None

        # find all relay directories
        relay_dirs = []
        for name in os.listdir(self.base_dir):
            if name.startswith("relay_"):
                full = os.path.join(self.base_dir, name)
                if os.path.isdir(full):
                    meta_path = os.path.join(full, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        relay_dirs.append((meta.get("round", 0), full))

        if not relay_dirs:
            return None

        # sort by round count (descending) and load the latest
        relay_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_dir = relay_dirs[0][1]

        # load directly from the relay directory (not a subdirectory)
        ckpt_mgr = CheckpointManager(os.path.dirname(latest_dir), version=self.version)
        ckpt_mgr.relay_dir = latest_dir
        return ckpt_mgr.load_relay()

    def load_relay_stage(self, round_id: int) -> Optional[Dict[str, Any]]:
        """Load a specific relay stage."""
        stage_dir = self.stage_dirs.get(round_id)
        if stage_dir is None:
            # try to find it
            stage_name = f"relay_{self.version}_stage_{round_id}"
            stage_dir = os.path.join(self.base_dir, stage_name)

        if not os.path.exists(stage_dir):
            return None

        ckpt_mgr = CheckpointManager(stage_dir, version=self.version)
        return ckpt_mgr.load_relay()

    def get_lineage(self) -> RelayLineage:
        """Get the current lineage."""
        return self.lineage

    def set_lineage(self, lineage: RelayLineage) -> None:
        """Set lineage (when continuing from a previous relay)."""
        self.lineage = lineage

    def available_stages(self) -> List[Dict[str, Any]]:
        """List available relay stages."""
        stages = []
        if not os.path.exists(self.base_dir):
            return stages

        for name in sorted(os.listdir(self.base_dir)):
            if name.startswith("relay_"):
                full = os.path.join(self.base_dir, name)
                meta_path = os.path.join(full, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    stages.append({
                        "name": name,
                        "round": meta.get("round", 0),
                        "version": meta.get("version", ""),
                    })
        return stages

    def summary(self) -> Dict[str, Any]:
        """Summary of relay state."""
        return {
            "version": self.version,
            "base_dir": self.base_dir,
            "lineage": self.lineage.to_dict(),
            "n_stages": len(self.stage_dirs),
            "available_stages": self.available_stages(),
        }
