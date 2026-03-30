#!/usr/bin/env -S python3 -u
"""
N-gram Boosting Eval for Parameter Golf.

Takes a trained model and boosts BPB by mixing neural predictions with
backward-looking N-gram cache (orders 2-7).

Key: only uses tokens ALREADY SCORED (backward-looking). Legal per competition rules.

Algorithm:
  For each chunk of val tokens:
    1. Score with neural model (record per-token NLL)
    2. Mix neural probs with N-gram probs → better predictions
    3. Add scored tokens to N-gram cache
    4. Next chunk benefits from larger cache

Mixing: entropy-adaptive (similar to PR #1026)
  High entropy (model uncertain) → trust N-gram more
  Low entropy (model confident) → trust neural more
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# CONFIG
# ==============================================================================
COMPUTE_DTYPE = mx.bfloat16
DATA_DIR = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

VOCAB_SIZE = 1024
NUM_LAYERS = 11
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
LOGIT_SOFTCAP = 30.0
SEQ_LEN = 1024
XSA_LAST_N = 4
BIGRAM_BUCKETS = 2048
BIGRAM_DIM = 128

# ==============================================================================
# MODEL (same as golf_v2)
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T

class RMSNormNoWeight(nn.Module):
    def __call__(self, x):
        return rms_norm(x)

class DualModeAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5
        self.use_xsa = use_xsa
    def _xsa(self, y, v):
        bsz, seqlen, dim = y.shape
        hd = self.head_dim; nkv = self.num_kv_heads; nh = self.num_heads; group = nh // nkv
        y_g = y.reshape(bsz, seqlen, nkv, group, hd)
        v_t = v.transpose(0, 2, 1, 3)
        vn = v_t / (mx.sqrt(mx.sum(v_t * v_t, axis=-1, keepdims=True)) + 1e-8)
        vn = mx.expand_dims(vn, axis=3)
        proj = mx.sum(y_g * vn, axis=-1, keepdims=True) * vn
        return (y_g - proj).reshape(bsz, seqlen, dim)
    def __call__(self, x, is_causal=True):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        if is_causal:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        if self.use_xsa:
            y = self._xsa(y, v)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mlp_mult)
        self.proj = CastedLinear(dim * mlp_mult, dim)
    def __call__(self, x):
        h = self.fc(x)
        h = mx.where(h >= 0, h, 0.5 * h)
        return self.proj(h * h)

class BigramHashEmbedding(nn.Module):
    def __init__(self, buckets, bigram_dim, model_dim):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, bigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim)
        self.scale = mx.array(0.05, dtype=mx.float32)
    def bigram_hash(self, tokens):
        t = tokens.astype(mx.int32); mod = self.buckets - 1
        shifted = mx.concatenate([mx.full((t.shape[0], 1), mod, dtype=mx.int32), t[:, :-1]], axis=1)
        return (36313 * t + 27191 * shifted) % mod
    def __call__(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        return self.proj(h) * self.scale.astype(h.dtype)

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)
    def __call__(self, x):
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx=0, use_xsa=False):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = DualModeAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))
    def __call__(self, x, x0, is_causal=True):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale, is_causal=is_causal)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x

class GPTv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.bigram = BigramHashEmbedding(BIGRAM_BUCKETS, BIGRAM_DIM, MODEL_DIM)
        self.smear = SmearGate(MODEL_DIM)
        self.num_encoder_layers = NUM_LAYERS // 2
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, MODEL_DIM), dtype=mx.float32)
        self.blocks = []
        for i in range(NUM_LAYERS):
            use_xsa = i >= (NUM_LAYERS - XSA_LAST_N)
            self.blocks.append(Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT,
                                     layer_idx=i, use_xsa=use_xsa))
        self.final_norm = RMSNormNoWeight()
    def softcap(self, logits):
        c = self.logit_softcap; return c * mx.tanh(logits / c)
    def forward_hidden(self, input_ids, is_causal=True):
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        x = x + self.bigram(input_ids).astype(COMPUTE_DTYPE)
        x = rms_norm(x); x = self.smear(x); x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, is_causal=is_causal); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, is_causal=is_causal)
        return self.final_norm(x)
    def get_logits(self, input_ids):
        h = self.forward_hidden(input_ids, is_causal=True)
        return self.softcap(h @ self.tok_emb.weight.astype(h.dtype).T)

# ==============================================================================
# DATA + TOKENIZER
# ==============================================================================
def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes).astype(np.int32, copy=False)

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True; piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut

# ==============================================================================
# N-GRAM CACHE (orders 2-7, backward-looking)
# ==============================================================================
class NgramCache:
    def __init__(self, max_order=7, vocab_size=VOCAB_SIZE):
        self.max_order = max_order
        self.vocab_size = vocab_size
        # For each order n, store counts: context_tuple -> count_array[vocab_size]
        self.counts = {}
        for n in range(2, max_order + 1):
            self.counts[n] = defaultdict(lambda: np.zeros(vocab_size, dtype=np.float32))
        self.total_added = 0

    def update(self, tokens):
        """Add scored tokens to cache (backward-looking)."""
        for n in range(2, self.max_order + 1):
            for i in range(n - 1, len(tokens)):
                ctx = tuple(tokens[i - n + 1: i])
                self.counts[n][ctx][tokens[i]] += 1
        self.total_added += len(tokens)

    def predict(self, context):
        """Get N-gram probability with multi-order backoff."""
        context = list(context)
        # Start with uniform
        probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        for n in range(2, self.max_order + 1):
            if len(context) < n - 1:
                continue
            ctx = tuple(context[-(n - 1):])
            if ctx not in self.counts[n]:
                continue
            counts = self.counts[n][ctx]
            total = counts.sum()
            if total < 1:
                continue
            # Kneser-Ney-like smoothing
            discount = 0.75
            smoothed = np.maximum(counts - discount, 0)
            num_nonzero = np.sum(counts > 0)
            backoff_mass = discount * num_nonzero / total
            ngram_probs = smoothed / total + backoff_mass * probs
            # Interpolation weight based on evidence
            weight = min(1.0, total / (total + 5.0 * n))
            probs = (1 - weight) * probs + weight * ngram_probs

        probs = np.maximum(probs, 1e-20)
        probs /= probs.sum()
        return probs

# ==============================================================================
# EVAL
# ==============================================================================
def eval_with_ngram(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                    is_boundary_token_lut, ngram_cache, max_tokens, chunk_seqs=16):
    """Score val tokens with neural + N-gram mixture.

    Process chunk by chunk:
      1. Get neural logits for chunk
      2. For each token: mix neural + N-gram probs (entropy-adaptive)
      3. Record NLL
      4. Add chunk tokens to N-gram cache
    """
    total_nll_bits = 0.0
    total_bytes = 0.0
    total_tok = 0
    n_tokens = min(val_tokens.size - 1, max_tokens)
    total_seqs = n_tokens // SEQ_LEN

    for s in range(0, total_seqs, chunk_seqs):
        e = min(s + chunk_seqs, total_seqs)
        chunk = val_tokens[s * SEQ_LEN: e * SEQ_LEN + 1]
        x_np = chunk[:-1].reshape(-1, SEQ_LEN)
        y_np = chunk[1:].reshape(-1, SEQ_LEN)

        # Neural forward pass
        x = mx.array(x_np, dtype=mx.int32)
        logits = model.get_logits(x).astype(mx.float32)
        neural_probs = mx.softmax(logits, axis=-1)
        mx.eval(neural_probs)
        neural_probs_np = np.array(neural_probs).reshape(-1, VOCAB_SIZE)

        flat_x = x_np.reshape(-1)
        flat_y = y_np.reshape(-1)
        n_tok_chunk = len(flat_y)

        for i in range(n_tok_chunk):
            tgt = int(flat_y[i])
            prev = int(flat_x[i])

            # Neural probability
            p_neural = neural_probs_np[i].astype(np.float64)
            p_neural = np.maximum(p_neural, 1e-20)

            # Neural entropy (for adaptive mixing)
            H = -np.sum(p_neural * np.log(p_neural + 1e-20))

            # N-gram probability
            # Context: tokens leading up to this position
            global_pos = s * SEQ_LEN + i
            ctx_start = max(0, global_pos - 6)  # max 6 context tokens for 7-gram
            context = val_tokens[ctx_start: global_pos + 1].tolist()
            p_ngram = ngram_cache.predict(context)

            # Entropy-adaptive mixing (similar to PR #1026)
            # High H (uncertain) → more N-gram; Low H (confident) → more neural
            alpha = 0.05 + 0.55 * (1.0 / (1.0 + math.exp(-2.0 * (H - 4.0))))
            # alpha ∈ [0.05, 0.60]: neural weight = 1-alpha ∈ [0.40, 0.95]

            # Mix
            p_mixed = (1 - alpha) * p_neural + alpha * p_ngram
            p_mixed = np.maximum(p_mixed, 1e-20)
            p_mixed /= p_mixed.sum()

            # NLL in bits
            nll_bits = -math.log2(p_mixed[tgt])

            # Bytes
            b = float(base_bytes_lut[tgt])
            if has_leading_space_lut[tgt] and not is_boundary_token_lut[prev]:
                b += 1.0

            total_nll_bits += nll_bits
            total_bytes += b
            total_tok += 1

        # Add scored tokens to N-gram cache (backward-looking: already scored)
        scored = val_tokens[s * SEQ_LEN: e * SEQ_LEN + 1].tolist()
        ngram_cache.update(scored)

        # Progress
        if total_tok % (chunk_seqs * SEQ_LEN) == 0 or s + chunk_seqs >= total_seqs:
            bpb = total_nll_bits / max(total_bytes, 1)
            cache_size = ngram_cache.total_added
            print(f"  [{total_tok:,}/{n_tokens:,}] BPB={bpb:.4f} | cache={cache_size:,} tokens")

    bpb = total_nll_bits / max(total_bytes, 1)
    return bpb


def eval_standard(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                  is_boundary_token_lut, max_tokens):
    """Standard causal BPB (no N-gram, for comparison)."""
    total_nll = 0.0
    total_tok = 0
    total_bytes = 0.0
    n_tokens = min(val_tokens.size - 1, max_tokens)
    total_seqs = n_tokens // SEQ_LEN

    for s in range(0, total_seqs, 16):
        e = min(s + 16, total_seqs)
        chunk = val_tokens[s * SEQ_LEN: e * SEQ_LEN + 1]
        x_np = chunk[:-1].reshape(-1, SEQ_LEN)
        y_np = chunk[1:].reshape(-1, SEQ_LEN)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        logits = model.get_logits(x).astype(mx.float32)
        logits_2d = logits.reshape(-1, VOCAB_SIZE)
        y_flat = y.reshape(-1)
        per_token = nn.losses.cross_entropy(logits_2d, y_flat, reduction="none")
        mx.eval(per_token)
        total_nll += float(mx.sum(per_token).item())
        ct = int(y_flat.size)
        total_tok += ct
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.float64)
        total_bytes += bytes_np.sum()

    avg_loss = total_nll / total_tok
    bpt = avg_loss / math.log(2.0)
    return bpt * (total_tok / total_bytes)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2_000_000,
                        help="Val tokens to eval (more = better N-gram cache)")
    parser.add_argument("--ngram_order", type=int, default=7)
    args = parser.parse_args()

    print("=" * 70)
    print("N-gram Boosting Eval")
    print(f"Max order: {args.ngram_order} | Max tokens: {args.max_tokens:,}")
    print("=" * 70)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE)

    # Validation
    val_tokens = load_validation_tokens(f"{DATA_DIR}/fineweb_val_*.bin", SEQ_LEN)
    print(f"Val tokens available: {val_tokens.size - 1:,}")

    # Model
    mx.random.seed(1337)
    model = GPTv2()
    weights = dict(np.load(args.model_path))
    model_keys = set(k for k, _ in tree_flatten(model.parameters()))
    mlx_weights = {}
    for k, v in weights.items():
        if k not in model_keys: continue
        if v.dtype.str == '|V2':
            mlx_weights[k] = mx.array(v.view(np.uint16)).view(mx.bfloat16)
        else:
            mlx_weights[k] = mx.array(v)
    model.update(tree_unflatten(list(mlx_weights.items())))
    mx.eval(model.parameters())
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model: {n_params:,} params from {args.model_path}")
    print()

    # 1. Standard eval (baseline)
    print("[1/2] Standard causal BPB...")
    t0 = time.time()
    standard_bpb = eval_standard(model, val_tokens, base_bytes_lut,
                                  has_leading_space_lut, is_boundary_token_lut,
                                  max_tokens=args.max_tokens)
    t1 = time.time()
    print(f"  Standard BPB = {standard_bpb:.4f}  ({t1-t0:.1f}s)")
    print()

    # 2. N-gram boosted eval
    print(f"[2/2] N-gram boosted BPB (orders 2-{args.ngram_order})...")
    ngram_cache = NgramCache(max_order=args.ngram_order, vocab_size=VOCAB_SIZE)
    t0 = time.time()
    ngram_bpb = eval_with_ngram(model, val_tokens, base_bytes_lut,
                                 has_leading_space_lut, is_boundary_token_lut,
                                 ngram_cache, max_tokens=args.max_tokens)
    t1 = time.time()
    print(f"  N-gram BPB = {ngram_bpb:.4f}  ({t1-t0:.1f}s)")
    print()

    # Summary
    delta = ngram_bpb - standard_bpb
    pct = delta / standard_bpb * 100
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Standard BPB:    {standard_bpb:.4f}")
    print(f"  N-gram BPB:      {ngram_bpb:.4f}  ({delta:+.4f}, {pct:+.1f}%)")
    print(f"  N-gram cache:    {ngram_cache.total_added:,} tokens")
    print(f"  Eval tokens:     {args.max_tokens:,}")
    if delta < 0:
        print(f"  N-gram WINS by {-delta:.4f} BPB!")
    else:
        print(f"  N-gram didn't help (need more eval tokens?)")
    print("=" * 70)


if __name__ == "__main__":
    main()
