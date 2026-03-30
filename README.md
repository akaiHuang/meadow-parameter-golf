# Meadow: Retrodiction-Augmented Language Model for Parameter Golf

**Author:** Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang))

## Approach: Retrodiction Training

We introduce **Retrodiction**, a novel auxiliary training loss inspired by the Petz recovery map from quantum information theory. The model trains on both forward and reversed sequences simultaneously, learning bidirectional representations while maintaining causal (left-to-right) attention.

### Core Idea

Standard AR training: predict next token from left context only.

Retrodiction training: **additionally** train on reversed sequences, so the model learns right-to-left patterns through the same causal attention mechanism. This enriches token embeddings with bidirectional information without requiring bidirectional attention.

```
Forward:  "the cat sat on the mat" → predict left-to-right
Reversed: "mat the on sat cat the" → predict left-to-right (= right-to-left in original)

loss = AR_loss(forward) + 0.3 * AR_loss(reversed)
```

### Theoretical Foundation

The Petz recovery map in quantum information theory provides the optimal retrodiction (inferring past from future). Our retrodiction loss is a direct application: by training the model to predict in reverse, we implement retrodiction at the language level.

Key result: **Retrodiction improves BPB by up to 3.6% at early training (step 200), stabilizing at ~0.6% at step 500**, compared to pure AR training at the same token count.

### Architecture

Upgraded beyond baseline:
- **16 layers, 512 dim, 3x MLP (39M params)** — larger than baseline 9L/27M
- Muon optimizer + AdamW (embeddings/scalars)
- EMA (decay=0.997, delayed start at 80%)
- XSA on last 4 layers (exclusive self-attention)
- BigramHash (2048 buckets) + SmearGate
- LeakyReLU(0.5)^2 activation
- **Int6 quantization + lzma compression → 14.8MB (within 16MB limit)**

### Eval Enhancements

- **N-gram boosting**: Backward-looking N-gram cache (orders 2-7) with entropy-adaptive mixing
- Standard causal BPB evaluation on FineWeb validation set

### Results (M1 Max, 64GB, local validation)

**All experiments: same architecture, same data (FineWeb), same tokenizer (SP1024), same eval method.**

#### Retrodiction vs Pure AR (same token count, fair comparison)

| Step | Tokens | Retro BPB | Pure AR BPB | Delta | Improvement |
|------|--------|-----------|-------------|-------|-------------|
| 100 | 7M | 2.155 | 2.183 | -0.028 | -1.3% |
| 200 | 13M | 1.934 | 2.006 | -0.072 | -3.6% |
| 400 | 26M | 1.727 | 1.764 | -0.037 | -2.1% |
| 500 | 33M | 1.714 | ~1.72 | -0.010 | -0.6% |
| 3500 | 229M | 1.464 | - | - | - |
| 4500 | 295M | 1.408 | - | - | - |
| 5000 | 328M | **1.3923** | - | 8.4 hrs on M1 (11L, 27M) |

#### 16-Layer 39M Model (fits in 16MB with Int6)

| Step | Tokens | 16L BPB | 11L BPB | Delta |
|------|--------|---------|---------|-------|
| 100 | 7M | 2.161 | 2.155 | +0.006 (larger model, slower start) |
| 500 | 33M | 1.705 | 1.714 | -0.009 (catching up) |
| 1000 | 66M | 1.576 | ~1.60 | -0.024 (pulling ahead) |
| 2000 | 131M | **1.508** | 1.517 | **-0.009** (16L wins) |

Quantization: 39M params × Int6 + lzma = **14.8MB** (within 16MB limit) - |

#### Other methods tested (step 400, 26M tokens)

| Method | BPB | vs Pure AR | Notes |
|--------|-----|------------|-------|
| Pure AR | 1.764 | baseline | Standard next-token prediction |
| CDM rightmask | 1.744 | -0.021 | Mask right-side tokens, predict from left |
| Retrodiction | 1.727 | -0.037 | Reversed sequence auxiliary loss |
| Retro + CDM-R alternating | 1.726* | -0.038* | *step 500 only |
| CDM random mask | 1.763 | -0.001 | Random positions masked |
| Petz-weighted loss | 2.091 | +0.327 | e^{Σ/2} weighting, too aggressive |

#### N-gram eval boosting (applied post-training)

| Eval tokens | Standard BPB | N-gram BPB | Improvement |
|-------------|-------------|------------|-------------|
| 2M | 1.520 | 1.522 | +0.002 (cache too small) |
| 14M | 1.520 | 1.502 | -0.018 |
| 23M | 1.520 | 1.487 | -0.033 |
| 46M | 1.520 | 1.477 | -0.043 |

N-gram improvement increases with eval size as cache accumulates more patterns.

Note: M1 uses 65K tokens/step. H100 uses 786K tokens/step (12x larger batch). H100 scores will be significantly better due to more total tokens in 10 minutes.

### Estimated H100 Performance (10 min, 8xH100)

Preliminary estimates based on M1 scaling trends. Requires H100 validation.

```
Pure AR baseline:              ~1.12 BPB (extrapolated)
Retrodiction (every 4 steps):  ~1.11 BPB (extrapolated)
+ N-gram eval:                 ~1.07 BPB (extrapolated)
Current #1:                     1.1194 BPB
```

### How to Run

```bash
# Training (5000 steps on M1, ~10 hours)
python3 -u train_retrodiction.py --steps 5000 --grad_accum 2 \
  --microbatch_tokens 32768 --max_sub_chunk 8192 \
  --warmdown 1250 --val_every 500 --val_tokens 1000000 \
  --save_path model_5000step.npz

# N-gram evaluation
python3 -u eval_ngram.py --model_path model_5000step.npz --max_tokens 60000000
```

### Novel Contributions

1. **Retrodiction training**: To our knowledge, the first application of Petz recovery map-inspired reversed sequence auxiliary loss in LLM training
2. **N-gram boosting**: Entropy-adaptive backward-looking N-gram cache for eval-time improvement
3. **Theoretical grounding**: Training method inspired by quantum information theory (Petz recovery theorem, Petz 1986)

### Project: Meadow

This submission is part of the **Meadow** project — an efficient language model research initiative exploring novel training methods grounded in information theory.

### License

MIT
