# Long-Range Relay Learning

## Overview

Relay learning is the cross-version continuation mechanism in DSCNS Phase 6. It enables studying whether a DSCNS that has been self-modifying for a long time can continue to learn effectively.

The core idea: instead of always starting experiments from a fresh base model, relay checkpoints capture the COMPLETE training state at round 450, allowing the next version to continue from where the previous version left off.

```
Relay_0 → 450 rounds → Relay_1 → 450 rounds → Relay_2 → ...
```

## What Is Relay Learning?

Relay learning addresses a fundamental question about self-modifying systems:

> Can a system that has been continuously adapting for a long time still learn new information, or has it become locked into its current configuration?

Standard experiments always start from a fresh base model (GPT-2 124M) with randomly initialized LoRA adapters. This gives us a clean baseline but doesn't tell us what happens after extensive self-modification.

Relay experiments answer this by:
1. Training a DSCNS for 450 rounds with full self-modification
2. Saving the complete training state as a "relay checkpoint"
3. Starting a new experiment FROM that checkpoint
4. Running for another 450 rounds
5. Repeating the process

This creates a chain of continued learning that can span thousands of effective rounds.

## Relay Checkpoint Format

Each relay checkpoint captures the complete state needed for seamless continuation:

```
relay/
├── model_state.pt          # LoRA adapter weights (all networks)
├── policy_state.pt         # PolicyAdapter weights + optimizer state
├── optimizer_state.pt      # Task-learning optimizer state
├── memory_snapshot.pt      # ModificationMemory contents
├── experience_value.pt     # ExperienceValueModel state
├── round_counter.json      # Current round number
├── random_state.pt         # Python + NumPy + Torch RNG state
├── config.yaml             # Experiment configuration
├── architecture.json       # Network topology + connections
├── metrics.json            # Performance history + evolution log
└── lineage.json            # Source/version tracking metadata
```

### Component Details

| Component | Description | Why It Matters |
|-----------|-------------|----------------|
| `model_state.pt` | LoRA adapter weights for all cognitive networks | Preserves learned specializations |
| `policy_state.pt` | PolicyAdapter neural network + optimizer | Continues experience-conditioned decision making |
| `optimizer_state.pt` | Task-learning optimizer (AdamW) | Maintains momentum for stable training |
| `memory_snapshot.pt` | ModificationMemory with past modification records | Prevents repeating known failures |
| `experience_value.pt` | ExperienceValueModel with value assignments | Preserves learned experience valuations |
| `round_counter.json` | Current round number (e.g., 450) | Ensures correct round sequencing |
| `random_state.pt` | RNG states for reproducibility | Enables exact replay of training trajectory |
| `config.yaml` | All hyperparameters | Ensures consistent training setup |
| `architecture.json` | Network IDs, connections, domain assignments | Preserves evolved topology |
| `metrics.json` | Performance history, evolution log | Enables trajectory analysis |
| `lineage.json` | Source version, target version, total rounds | Tracks the relay chain |

## Lineage Tracking

Every relay checkpoint records its lineage:

```json
{
  "source_version": "v0.6.0",
  "source_relay": "relay_v0.6.0",
  "target_version": "v0.6.0",
  "continued_rounds": 450,
  "total_lineage_rounds": 900,
  "source_condition": "FullPolicy",
  "source_seed": 42,
  "created_at": "2024-01-15 14:30:00"
}
```

Key fields:
- `source_version`: The DSCNS version that created this relay
- `source_relay`: Which relay checkpoint this was loaded from (empty for Relay_0)
- `target_version`: The version that will continue from this relay
- `continued_rounds`: How many rounds were run after loading this relay
- `total_lineage_rounds`: Cumulative rounds across all relay generations
- `source_condition`: The experimental condition that produced this relay
- `source_seed`: The random seed used in the source experiment

## Standard vs. Relay Experiments

### Standard Experiments (Fresh Init)

```
Start: Base model (GPT-2) + fresh LoRA adapters
Rounds: 0 → 450
Checkpoint: best/final (model weights only)
```

Standard experiments provide the baseline: how does a DSCNS learn when starting from scratch?

### Relay Experiments (Continued)

```
Start: Relay checkpoint from previous experiment
Rounds: 450 → 900
Checkpoint: best/final + relay (full state)
```

Relay experiments show: does continued self-modification help or hurt learning?

### Key Differences

| Aspect | Standard | Relay |
|--------|----------|-------|
| Starting state | Fresh base model | Previous relay checkpoint |
| Effective round 0 | True round 0 | Round 450 (or N×450) |
| Initial LoRA weights | Random | Trained |
| Initial policy state | Random | Trained |
| Initial memory | Empty | Populated |
| Interpretation | Learning from scratch | Continued learning |

## Important Caveat: Continuity, Not Improvement

**Relay demonstrates continuity, NOT guaranteed improvement.**

A relay experiment showing maintained or improved performance proves that the system *can* continue learning. A relay experiment showing degraded performance doesn't mean the system is broken—it may indicate:

1. **Catastrophic interference**: New learning overwriting old knowledge
2. **Plasticity collapse**: The system has become too specialized
3. **Memory saturation**: Past experiences are crowding out new ones
4. **Policy ossification**: The modification policy has converged prematurely

The goal is to study these dynamics, not to guarantee better performance.

## Current Relay Status

Phase 6 experiments include relay checkpoints at round 450:

- **11 conditions**: FullPolicy, NoMemory, FrozenPolicy, RandomMemory, ZeroMemory, NoCredit, NoAlternatives, NoExploration, NoOutcomeReward, Oracle, Random
- **5 seeds per condition**: Ensuring statistical robustness
- **Total**: 55 relay checkpoints (11 × 5)

Each checkpoint is saved at the end of the standard 450-round experiment and can be loaded for continuation.

## Loading and Continuing from a Relay Checkpoint

### Step 1: Load the Relay

```python
from dscns.relay_manager import RelayManager
from dscns.config import DSCNSConfig

# Initialize relay manager
relay_mgr = RelayManager(
    base_dir="experiments/relay/FullPolicy/seed_42",
    version="v0.6.0"
)

# Load the latest relay checkpoint
relay_state = relay_mgr.load_latest_relay()

if relay_state is not None:
    print(f"Loaded relay from round {relay_state['round_counter']}")
    print(f"Lineage: {relay_state['lineage']}")
```

### Step 2: Restore State

```python
from dscns.system import DSCNSSystem

# Create system with relay state
system = DSCNSSystem(base_model, config, domain_exemplars, probe_sets)

# Restore model weights
for net_id, net in system.networks.items():
    if net_id in relay_state['model_state']:
        net.restore_adapter(relay_state['model_state'][net_id])

# Restore policy
system.self_mod.policy.load_state_dict(relay_state['policy_state'])

# Restore memory
system.self_mod.memory.load_state_dict(relay_state['memory_snapshot'])

# Restore experience values
system.self_mod.experience_value.load_state_dict(relay_state['experience_value'])

# Restore round counter
system.round_idx = relay_state['round_counter']['round']
```

### Step 3: Continue Training

```python
# Continue from where we left off
for round_id in range(system.round_idx, system.round_idx + 450):
    # Standard training loop
    experiences = data_stream.sample(batch_size)
    result = system.process_experiences(experiences)
    
    # Save relay at round 900 (450 rounds after relay at 450)
    if round_id == 900:
        relay_mgr.save_relay(
            relay_state=collect_relay_state(system),
            condition="FullPolicy",
            seed=42,
            round_id=round_id,
        )
```

### Step 4: Verify Lineage

```python
# Check the lineage chain
lineage = relay_state['lineage']
print(f"This relay continues from round {lineage['continued_rounds']}")
print(f"Total lineage rounds: {lineage['total_lineage_rounds']}")
print(f"Source: {lineage['source_version']} ({lineage['source_condition']})")
```

## Relay Directory Structure

```
experiments/
├── phase6/
│   ├── FullPolicy/
│   │   ├── seed_42/
│   │   │   ├── best/           # Best checkpoint
│   │   │   ├── final/          # Final checkpoint
│   │   │   ├── relay/          # Relay checkpoint (round 450)
│   │   │   └── raw/            # Per-round metrics
│   │   ├── seed_123/
│   │   └── ...
│   ├── NoMemory/
│   └── ...
├── relay/
│   ├── FullPolicy/
│   │   ├── seed_42/
│   │   │   ├── relay_v0.6.0_stage_450/
│   │   │   ├── relay_v0.6.0_stage_900/
│   │   │   ├── relay_v0.6.0_stage_1350/
│   │   │   └── relay_v0.6.0_stage_1800/
│   │   └── ...
│   └── ...
```

## Analysis Questions

Relay experiments enable studying:

1. **Continuity**: Does performance degrade, maintain, or improve across relay generations?
2. **Plasticity**: Does the system become more or less adaptable over time?
3. **Memory effects**: How does accumulated experience affect new learning?
4. **Policy evolution**: Does the modification policy converge or continue adapting?
5. **Topology stability**: Does the network structure stabilize or keep evolving?

## Implementation Details

### RelayManager

The `RelayManager` class handles:
- Saving relay checkpoints with full state
- Loading relay checkpoints (latest or specific stage)
- Tracking lineage across relay generations
- Managing stage directories for multi-stage relays

### CheckpointManager

The `CheckpointManager` class handles:
- Saving best/final/relay checkpoints
- Computing and verifying SHA256 checksums
- Managing checkpoint metadata
- Generating artifact manifests

### State Collection

When saving a relay, the system collects:
```python
relay_state = {
    'model_state': {net_id: net.snapshot_adapter() for net_id, net in systems.networks.items()},
    'policy_state': system.self_mod.policy.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'memory_snapshot': system.self_mod.memory.state_dict(),
    'experience_value': system.self_mod.experience_value.state_dict(),
    'round_counter': {'round': system.round_idx},
    'random_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    },
    'config': config.to_dict(),
    'architecture': {
        'networks': {nid: {'domain': net.domain, 'name': net.name} for nid, net in system.networks.items()},
        'connections': dict(system.connections),
    },
    'metrics': {
        'performance_history': system.eval_history,
        'evolution_log': system.self_mod.evolution_log,
    },
}
```

## Best Practices

1. **Always check lineage**: Understand what round you're continuing from
2. **Verify checkpoint integrity**: Use SHA256 verification before loading
3. **Match configurations**: Ensure relay config matches continuation config
4. **Track round numbers**: Account for the offset (round 450 becomes round 0 in the new run)
5. **Save intermediate relays**: Consider saving at rounds 900, 1350, 1800 for finer-grained analysis

## References

- DSCNS Design Report (docs/DSCNS_design_report.md)
- Checkpoint Manager (dscns/checkpoint_manager.py)
- Relay Manager (dscns/relay_manager.py)
- Phase 6 Experiment Configuration (dscns/config.py)
