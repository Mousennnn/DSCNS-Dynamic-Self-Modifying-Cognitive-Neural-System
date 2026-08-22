# Inference Pipeline

## Overview

The DSCNS inference pipeline provides tools for loading trained models and running inference on new inputs. It supports three modes: baseline (no self-modification), self-modify (standard trained model), and relay (continued learning from relay checkpoints).

## Scripts

### scripts/infer.py

The main inference script for running predictions with trained DSCNS models.

#### Usage

```bash
python scripts/infer.py --mode <mode> --checkpoint <path> --input <text>
```

#### Modes

**Baseline Mode**
```bash
python scripts/infer.py --mode baseline --input "What is 2 + 2?"
```
- Uses the base GPT-2 model without any LoRA adapters
- No self-modification applied
- Serves as the control for comparing trained models
- Fastest inference (no adapter overhead)

**Self-Modify Mode**
```bash
python scripts/infer.py --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/best/model.pt \
  --input "Solve for x: 2x + 3 = 7"
```
- Loads the trained DSCNS with all cognitive networks
- Applies the learned modification policy
- Uses experience-conditioned decision making
- Most comprehensive inference mode

**Relay Mode**
```bash
python scripts/infer.py --mode relay \
  --checkpoint experiments/relay/FullPolicy/seed_42/relay_v0.6.0_stage_900/ \
  --input "Explain the concept of recursion"
```
- Loads a relay checkpoint (full state)
- Continues from where the relay left off
- Includes accumulated experience and memory
- Useful for studying long-horizon learning effects

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Inference mode: baseline, self-modify, relay | Required |
| `--checkpoint` | Path to model checkpoint or relay directory | Required for self-modify/relay |
| `--input` | Input text for inference | Required |
| `--max-length` | Maximum generation length | 256 |
| `--temperature` | Sampling temperature | 0.8 |
| `--top-k` | Top-k sampling | 50 |
| `--device` | Device to use (cuda/cpu) | cuda |
| `--output-format` | Output format: text, json, detailed | text |

#### Output Formats

**Text Format** (default)
```
Input: What is 2 + 2?
Output: 4
Confidence: 0.95
```

**JSON Format**
```json
{
  "input": "What is 2 + 2?",
  "output": "4",
  "confidence": 0.95,
  "network_activations": {"N1": 0.8, "N2": 0.9, "N3": 0.7, "N4": 0.6, "N5": 0.85},
  "modification_events": [],
  "memory_retrievals": 2,
  "inference_time_ms": 45.2
}
```

**Detailed Format**
```
=== DSCNS Inference Report ===
Input: What is 2 + 2?

Network Activations:
  N1 (WorldKnowledge): 0.80 relevance, 0.75 confidence
  N2 (Math):           0.90 relevance, 0.85 confidence
  N3 (Logic):          0.70 relevance, 0.65 confidence
  N4 (Language):       0.60 relevance, 0.55 confidence
  N5 (Verification):   0.85 relevance, 0.80 confidence

Verification:
  Conflict detected: No
  Consensus score: 0.82
  Trust weights: {N1: 0.7, N2: 0.8, N3: 0.6, N4: 0.5, N5: 0.75}

Meta-Cognitive Decision:
  Action: internalize
  Target networks: [N2, N5]
  Internalization level: 0.85

Memory Retrieval:
  Similar experiences found: 2
  Top match: "2+2=4 arithmetic basic" (similarity: 0.92)

Output: 4
Confidence: 0.95
```

### scripts/demo_inference.py

An interactive demo script for exploring DSCNS inference capabilities.

#### Usage

```bash
python scripts/demo_inference.py
```

#### Features

1. **Interactive Mode**: Run inference on user inputs
2. **Batch Mode**: Process multiple inputs from a file
3. **Comparison Mode**: Compare baseline vs. trained model outputs
4. **Network Analysis**: Show which cognitive networks are activated
5. **Memory Inspection**: View retrieved experiences from memory
6. **Modification Tracking**: Show when and how the model modifies itself

#### Demo Commands

```
> infer What is the capital of France?
> compare The sky is blue
> networks
> memory
> modifications
> batch inputs.txt
> quit
```

## Checkpoint Selection

### Best Checkpoint

```bash
python scripts/infer.py --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/best/ \
  --input "..."
```

- Selected by validation score (no test leakage)
- Score = w_perf × Performance - w_rfr × RFR - w_drift × Drift + w_stab × Stability
- Typically the best generalization performance
- May not be the final state of training

### Final Checkpoint

```bash
python scripts/infer.py --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/final/ \
  --input "..."
```

- The state at the last training round (round 450)
- Always saved, may overwrite previous final
- Shows the end state of training
- May include recent overfitting

### Relay Checkpoint

```bash
python scripts/infer.py --mode relay \
  --checkpoint experiments/relay/FullPolicy/seed_42/relay_v0.6.0_stage_450/ \
  --input "..."
```

- Full training state for continuation
- Includes all components (model, policy, memory, etc.)
- Used for starting relay experiments
- Most comprehensive checkpoint

## Pipeline Trace

The inference pipeline trace shows the decision-making process:

```
[0.00ms] Input received: "What is 2 + 2?"
[0.12ms] Tokenization: 6 tokens
[1.23ms] Base model forward pass
[2.45ms] Network evaluation:
         N1: relevance=0.80, confidence=0.75
         N2: relevance=0.90, confidence=0.85
         N3: relevance=0.70, confidence=0.65
         N4: relevance=0.60, confidence=0.55
         N5: relevance=0.85, confidence=0.80
[3.67ms] Cross-network verification:
         Conflict: No
         Consensus: 0.82
[4.89ms] Meta-cognitive decision:
         Action: internalize
         Targets: [N2, N5]
[5.12ms] Memory retrieval:
         Query: "What is 2 + 2?"
         Matches: 2 (best: 0.92 similarity)
[6.34ms] Policy evaluation:
         Target: N2 (mathematical reasoning)
         Magnitude: 0.15
         Confidence: 0.88
[7.56ms] Generation:
         Token 1: "4" (prob=0.95)
[8.78ms] Post-processing:
         Confidence: 0.95
         Total time: 8.78ms
```

## Example Commands

### Basic Inference

```bash
# Baseline inference
python scripts/infer.py --mode baseline \
  --input "Explain quantum computing in simple terms"

# Self-modified model inference
python scripts/infer.py --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/best/ \
  --input "Explain quantum computing in simple terms"

# Relay model inference
python scripts/infer.py --mode relay \
  --checkpoint experiments/relay/FullPolicy/seed_42/relay_v0.6.0_stage_900/ \
  --input "Explain quantum computing in simple terms"
```

### Comparison

```bash
# Compare all three modes
python scripts/demo_inference.py --compare \
  --input "Solve: 3x - 7 = 14" \
  --baseline experiments/baseline/ \
  --self-modify experiments/phase6/FullPolicy/seed_42/best/ \
  --relay experiments/relay/FullPolicy/seed_42/relay_v0.6.0_stage_900/
```

### Batch Processing

```bash
# Process multiple inputs
python scripts/infer.py --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/best/ \
  --input-file inputs.txt \
  --output-file outputs.json \
  --output-format json
```

### Network Analysis

```bash
# Show detailed network activations
python scripts/demo_inference.py \
  --mode self-modify \
  --checkpoint experiments/phase6/FullPolicy/seed_42/best/ \
  --input "What is the derivative of x^2?" \
  --show-networks \
  --show-memory \
  --show-modifications
```

### Temperature Sweep

```bash
# Test different temperatures
for temp in 0.1 0.3 0.5 0.7 0.9; do
  python scripts/infer.py --mode self-modify \
    --checkpoint experiments/phase6/FullPolicy/seed_42/best/ \
    --input "Write a haiku about AI" \
    --temperature $temp \
    --output-file "outputs_temp_${temp}.json"
done
```

## Performance Considerations

### Inference Speed

| Mode | Typical Latency | Notes |
|------|----------------|-------|
| Baseline | ~5ms | No adapter overhead |
| Self-Modify | ~10ms | Adapter + policy evaluation |
| Relay | ~12ms | Full state loading |

### Memory Usage

| Mode | GPU Memory | Notes |
|------|------------|-------|
| Baseline | ~500MB | Base model only |
| Self-Modify | ~800MB | Base + adapters + policy |
| Relay | ~1GB | All components loaded |

### Optimization Tips

1. **Use baseline mode** when you don't need self-modification
2. **Batch multiple inputs** to amortize model loading
3. **Cache loaded models** if running many inferences
4. **Use GPU** for faster inference (default)
5. **Reduce max-length** for shorter generation

## Troubleshooting

### Checkpoint Not Found

```
Error: Checkpoint not found at experiments/phase6/FullPolicy/seed_42/best/
```

Verify the path exists and contains `model.pt` or `metadata.json`.

### Relay Loading Error

```
Error: Failed to load relay state
```

Check that the relay directory contains all required components (model_state.pt, policy_state.pt, etc.).

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

Try:
- Using `--device cpu`
- Reducing batch size
- Using a smaller model

### Missing Dependencies

```
ModuleNotFoundError: No module named 'dscns'
```

Ensure the DSCNS package is installed:
```bash
pip install -e .
```

## API Reference

### DSCNSInference

```python
from dscns.inference import DSCNSInference

# Initialize
inference = DSCNSInference(
    mode="self-modify",
    checkpoint_path="experiments/phase6/FullPolicy/seed_42/best/",
    device="cuda"
)

# Single inference
result = inference.run("What is 2 + 2?")
print(result.output)  # "4"
print(result.confidence)  # 0.95
print(result.network_activations)  # {N1: 0.8, N2: 0.9, ...}

# Batch inference
results = inference.batch(["Question 1", "Question 2", "Question 3"])

# Detailed trace
trace = inference.run_with_trace("Solve for x: 2x = 10")
for step in trace.steps:
    print(f"[{step.time_ms}ms] {step.description}")
```

### RelayLoader

```python
from dscns.relay_manager import RelayManager

# Load relay
relay_mgr = RelayManager(base_dir="experiments/relay/FullPolicy/seed_42/")
relay_state = relay_mgr.load_latest_relay()

# Load specific stage
relay_state = relay_mgr.load_relay_stage(round_id=900)

# List available stages
stages = relay_mgr.available_stages()
for stage in stages:
    print(f"Stage: {stage['name']}, Round: {stage['round']}")
```

## References

- Long-Range Relay Learning (docs/LONG_RANGE_RELAY.md)
- Checkpoint Manager (dscns/checkpoint_manager.py)
- Relay Manager (dscns/relay_manager.py)
- Inference Pipeline (scripts/infer.py)
- Demo Script (scripts/demo_inference.py)
