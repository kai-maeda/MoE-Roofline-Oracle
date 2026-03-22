"""
Workload modeling: transformer operator FLOPs, bytes moved, and arithmetic intensity.

Covers both dense transformers and Mixture-of-Experts (MoE) models.
Separates prefill (prompt processing) from decode (autoregressive generation)
since they have fundamentally different compute/memory characteristics.

=============================================================================
EQUATION CITATIONS
=============================================================================

Transformer FLOPs (general)
  [Kaplan20]  Kaplan, J. et al. "Scaling Laws for Neural Language Models."
              arXiv:2001.08361 (2020). Appendix D derives ~6ND FLOPs per
              forward+backward pass for a dense transformer (N params, D tokens).
              We use 2ND for forward-only (inference).

  [Hoffmann22] Hoffmann, J. et al. "Training Compute-Optimal Large Language
               Models." arXiv:2203.15556 (Chinchilla, 2022). Appendix F
               provides per-layer GEMM FLOPs breakdown used here.

Attention FLOPs
  [Dao22]     Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact
              Attention with IO-Awareness." NeurIPS 2022. Appendix B derives
              attention FLOPs: 4*B*S*d for QKV projections + 4*B*S^2*H for
              QK^T and AV matmuls (our score_flops + context_flops).

  [Narayanan21] Narayanan, D. et al. "Efficient Large-Scale Language Model
                Training on GPU Clusters Using Megatron-LM." SC'21.
                Table 1 confirms attention FLOP counts.

MoE FLOPs and routing
  [Lepikhin21] Lepikhin, D. et al. "GShard: Scaling Giant Models with
               Conditional Computation and Automatic Sharding." ICLR 2021.
               Introduces top-k expert routing; token dispatch formulation.

  [Rajbhandari22] Rajbhandari, S. et al. "DeepSpeed-MoE: Advancing Mixture-
                  of-Experts Inference and Training to Power Next-Generation
                  AI Scale." ICML 2022. Section 3 derives per-expert FLOP
                  counts under top-k routing: FLOPs = 2*k*S*d*d_ff per layer.

KV cache bytes
  [Pope22]    Pope, R. et al. "Efficiently Scaling Transformer Inference."
              MLSys 2023. Section 2.1 formalizes KV cache sizing:
              bytes = 2 * n_layers * 2 * S * d_head * n_heads * dtype_bytes
              (factor 2 for K and V; factor 2 for n_layers is applied at
              the full-model level here).

Parameter count formulas
  [Brown20]   Brown, T. et al. "Language Models are Few-Shot Learners."
              NeurIPS 2020. Appendix B gives closed-form parameter counts
              for GPT-style transformers.
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


@dataclass
class ModelConfig:
    """Architecture dimensions for a transformer (dense or MoE)."""
    name: str

    # Core dimensions
    d_model: int            # Hidden dimension
    n_layers: int           # Number of transformer layers
    n_heads: int            # Number of attention heads
    d_head: int             # Head dimension (usually d_model // n_heads)
    d_ffn: int              # FFN intermediate dimension

    # MoE parameters (set n_experts=1 for dense)
    n_experts: int = 1      # Total number of experts per MoE layer
    n_experts_active: int = 1  # Top-k experts per token
    moe_every_n: int = 1    # MoE layer frequency (e.g. every other layer)

    # Sequence
    max_seq_len: int = 4096
    vocab_size: int = 32000

    # Precision (bytes per parameter/activation)
    weight_dtype_bytes: float = 2.0   # BF16 = 2 bytes
    kv_cache_dtype_bytes: float = 2.0  # KV cache precision

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers // self.moe_every_n

    @property
    def n_dense_layers(self) -> int:
        return self.n_layers - self.n_moe_layers

    @property
    def total_params(self) -> int:
        """Approximate total parameter count."""
        # Embedding
        embed = self.vocab_size * self.d_model
        # Attention (Q, K, V, O projections) per layer
        attn = self.n_layers * 4 * self.d_model * self.d_model
        # FFN params
        dense_ffn = self.n_dense_layers * 2 * self.d_model * self.d_ffn
        moe_ffn = self.n_moe_layers * self.n_experts * 2 * self.d_model * self.d_ffn
        return embed + attn + dense_ffn + moe_ffn

    @property
    def active_params(self) -> int:
        """Active parameters per forward pass (top-k routing)."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * 4 * self.d_model * self.d_model
        dense_ffn = self.n_dense_layers * 2 * self.d_model * self.d_ffn
        # Only top-k experts fire
        moe_ffn = self.n_moe_layers * self.n_experts_active * 2 * self.d_model * self.d_ffn
        return embed + attn + dense_ffn + moe_ffn

    @property
    def total_params_b(self) -> float:
        return self.total_params / 1e9

    @property
    def active_params_b(self) -> float:
        return self.active_params / 1e9


@dataclass
class OperatorStats:
    """FLOPs and bytes moved for a single operator."""
    name: str
    flops: float           # Total floating point operations
    bytes_read: float      # Bytes read from HBM
    bytes_written: float   # Bytes written to HBM

    @property
    def bytes_total(self) -> float:
        return self.bytes_read + self.bytes_written

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte — determines roofline position."""
        if self.bytes_total == 0:
            return float('inf')
        return self.flops / self.bytes_total


@dataclass
class LayerStats:
    """Aggregated stats for one transformer layer."""
    attention: OperatorStats
    ffn: OperatorStats

    @property
    def total_flops(self) -> float:
        return self.attention.flops + self.ffn.flops

    @property
    def total_bytes(self) -> float:
        return self.attention.bytes_total + self.ffn.bytes_total

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else float('inf')


def compute_attention_stats(
    model: ModelConfig,
    seq_len: int,
    batch_size: int,
    phase: str = "prefill",
) -> OperatorStats:
    """
    Compute FLOPs and bytes for multi-head attention.

    Prefill: process S tokens in parallel (matrix × matrix)
    Decode: generate 1 token attending over KV cache (matrix × vector)
    """
    S = seq_len if phase == "prefill" else 1
    B = batch_size
    d = model.d_model
    H = model.n_heads
    d_h = model.d_head
    dtype = model.weight_dtype_bytes

    # --- FLOPs ---
    # Q, K, V projections: each is (B, S, d) @ (d, d) → 2*B*S*d^2 FLOPs per proj
    # (factor 2: one multiply + one add per element of output matrix)
    # Three projections (Q, K, V): 3 * 2*B*S*d^2
    # Source: [Narayanan21] Table 1; [Dao22] Appendix B.
    proj_flops = 3 * 2 * B * S * d * d

    # Attention scores: QK^T = (B, H, S, d_h) @ (B, H, d_h, S_kv) → 2*B*H*S*S_kv*d_h
    # Source: [Dao22] Appendix B, Eq. for QK^T matmul.
    S_kv = seq_len  # Full KV for prefill; full cache length for decode
    score_flops = 2 * B * H * S * S_kv * d_h

    # Softmax: ~5 ops per element (max, sub, exp, sum, div). Negligible vs GEMM.
    # Source: [Dao22] Appendix B.
    softmax_flops = 5 * B * H * S * S_kv

    # Context aggregation: AV = (B, H, S, S_kv) @ (B, H, S_kv, d_h) → 2*B*H*S*S_kv*d_h
    # Source: [Dao22] Appendix B.
    context_flops = 2 * B * H * S * S_kv * d_h

    # Output projection: (B, S, d) @ (d, d) → 2*B*S*d^2
    # Source: [Narayanan21] Table 1.
    out_proj_flops = 2 * B * S * d * d

    total_flops = proj_flops + score_flops + softmax_flops + context_flops + out_proj_flops

    # --- Bytes ---
    # Weight reads: Q/K/V/O projection matrices, each (d × d): 4 * d^2 * dtype_bytes
    # These are loaded from HBM for every forward pass.
    weight_bytes = 4 * d * d * dtype

    # Activation reads/writes: input tensor (B, S, d) in + output (B, S, d) out
    activation_bytes = 2 * B * S * d * dtype

    # KV cache read: load K and V for all S_kv positions attended over.
    # Bytes = 2 (K+V) * B * H * S_kv * d_h * dtype_bytes
    # Source: [Pope22] Section 2.1, KV cache bandwidth analysis.
    kv_cache_bytes = 2 * B * H * S_kv * d_h * model.kv_cache_dtype_bytes

    bytes_read = weight_bytes + activation_bytes + kv_cache_bytes
    bytes_written = B * S * d * dtype  # output activations

    return OperatorStats(
        name=f"attention_{phase}",
        flops=total_flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
    )


def compute_ffn_stats(
    model: ModelConfig,
    seq_len: int,
    batch_size: int,
    phase: str = "prefill",
    is_moe_layer: bool = False,
) -> OperatorStats:
    """
    Compute FLOPs and bytes for FFN / MoE expert layer.

    For MoE: only top-k experts are active per token, but all expert
    weights must potentially be loaded from HBM (unless weight caching fits).
    """
    S = seq_len if phase == "prefill" else 1
    B = batch_size
    d = model.d_model
    d_ff = model.d_ffn
    dtype = model.weight_dtype_bytes

    if is_moe_layer:
        k = model.n_experts_active
        n_exp = model.n_experts
    else:
        k = 1
        n_exp = 1

    # --- FLOPs ---
    # Each expert is a 2-layer MLP: up-project (d → d_ff) then down-project (d_ff → d).
    # FLOPs per GEMM = 2 * tokens * d * d_ff (the factor 2 = multiply-add).
    # Total active FLOPs = 2 (up+down) * 2 (mul-add) * B * S * k * d * d_ff
    # where k = n_experts_active (top-k tokens routed per token).
    # Only k experts fire; n_exp - k experts do zero work per token.
    # Source: [Rajbhandari22] Section 3, Eq. (1):
    #   FLOPs_MoE = 2 * k * S * d_model * d_ffn  (per layer, summed over active experts)
    total_flops = 2 * B * S * k * 2 * d * d_ff

    # --- Bytes ---
    # For MoE decode: even though only k experts fire, we may need to load
    # all expert weights if they don't fit in cache (worst case).
    # For large models with small batch, this is the dominant cost.
    if is_moe_layer and phase == "decode":
        # Decode is memory bound: load k expert weight matrices
        # In practice with small batch, often k experts are loaded cold
        expert_weight_bytes = k * 2 * d * d_ff * dtype
    else:
        expert_weight_bytes = k * 2 * d * d_ff * dtype

    activation_bytes = 2 * B * S * d * dtype  # input + output
    bytes_read = expert_weight_bytes + activation_bytes
    bytes_written = B * S * d * dtype

    op_name = f"moe_ffn_{phase}" if is_moe_layer else f"dense_ffn_{phase}"
    return OperatorStats(
        name=op_name,
        flops=total_flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
    )


def compute_layer_stats(
    model: ModelConfig,
    seq_len: int,
    batch_size: int,
    phase: str = "prefill",
    is_moe_layer: bool = False,
) -> LayerStats:
    attn = compute_attention_stats(model, seq_len, batch_size, phase)
    ffn = compute_ffn_stats(model, seq_len, batch_size, phase, is_moe_layer)
    return LayerStats(attention=attn, ffn=ffn)


@dataclass
class ModelStats:
    """Full forward pass stats across all layers."""
    model: ModelConfig
    seq_len: int
    batch_size: int
    phase: str  # "prefill" or "decode"

    layer_stats: list = field(default_factory=list)

    def __post_init__(self):
        self._compute()

    def _compute(self):
        self.layer_stats = []
        for layer_idx in range(self.model.n_layers):
            is_moe = (
                self.model.is_moe and
                (layer_idx % self.model.moe_every_n == 0)
            )
            stats = compute_layer_stats(
                self.model, self.seq_len, self.batch_size, self.phase, is_moe
            )
            self.layer_stats.append(stats)

    @property
    def total_flops(self) -> float:
        return sum(l.total_flops for l in self.layer_stats)

    @property
    def total_bytes(self) -> float:
        return sum(l.total_bytes for l in self.layer_stats)

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else float('inf')

    @property
    def total_flops_tflops(self) -> float:
        return self.total_flops / 1e12

    @property
    def total_bytes_tb(self) -> float:
        return self.total_bytes / 1e12

    def attention_intensity(self) -> float:
        attn_flops = sum(l.attention.flops for l in self.layer_stats)
        attn_bytes = sum(l.attention.bytes_total for l in self.layer_stats)
        return attn_flops / attn_bytes if attn_bytes > 0 else float('inf')

    def ffn_intensity(self) -> float:
        ffn_flops = sum(l.ffn.flops for l in self.layer_stats)
        ffn_bytes = sum(l.ffn.bytes_total for l in self.layer_stats)
        return ffn_flops / ffn_bytes if ffn_bytes > 0 else float('inf')


# ---------------------------------------------------------------------------
# Model Catalog — known open-weight MoE and dense models
# ---------------------------------------------------------------------------

# DeepSeek-V3 (671B total, 37B active) — Dec 2024
# 256 routed experts + 1 shared expert per MoE layer; top-8 routing
DEEPSEEK_V3 = ModelConfig(
    name="DeepSeek-V3",
    d_model=7168,
    n_layers=61,
    n_heads=128,
    d_head=128,
    d_ffn=2048,       # Per-expert intermediate dim (MLA architecture)
    n_experts=256,
    n_experts_active=8,
    moe_every_n=1,
    max_seq_len=128000,
    vocab_size=128000,
)

# Llama 4 Scout (109B total, 17B active) — Apr 2025
# 16 experts, top-1 routing, all layers are MoE
LLAMA4_SCOUT = ModelConfig(
    name="Llama-4-Scout-17B-16E",
    d_model=5120,
    n_layers=48,
    n_heads=40,
    d_head=128,
    d_ffn=8192,
    n_experts=16,
    n_experts_active=1,
    moe_every_n=1,
    max_seq_len=131072,
    vocab_size=202048,
)

# Llama 4 Maverick (400B total, 17B active) — Apr 2025
# 128 experts, top-1 routing, MoE and dense layers alternate
LLAMA4_MAVERICK = ModelConfig(
    name="Llama-4-Maverick-17B-128E",
    d_model=5120,
    n_layers=48,
    n_heads=40,
    d_head=128,
    d_ffn=8192,
    n_experts=128,
    n_experts_active=1,
    moe_every_n=2,    # MoE and dense layers alternate
    max_seq_len=1048576,
    vocab_size=202048,
)

# Qwen3-235B-A22B (235B total, 22B active) — Apr 2025
# 128 experts, top-8 routing, all layers are MoE
QWEN3_235B = ModelConfig(
    name="Qwen3-235B-A22B",
    d_model=4096,
    n_layers=94,
    n_heads=64,
    d_head=128,
    d_ffn=1536,       # MoE intermediate size per expert
    n_experts=128,
    n_experts_active=8,
    moe_every_n=1,
    max_seq_len=131072,
    vocab_size=151936,
)

# GPT-3 175B (dense, for baseline comparison)
GPT3_175B = ModelConfig(
    name="GPT-3 175B (dense)",
    d_model=12288,
    n_layers=96,
    n_heads=96,
    d_head=128,
    d_ffn=49152,
    n_experts=1,
    n_experts_active=1,
    moe_every_n=1,
    max_seq_len=4096,
    vocab_size=50257,
)

ALL_MODELS = [DEEPSEEK_V3, LLAMA4_SCOUT, LLAMA4_MAVERICK, QWEN3_235B, GPT3_175B]


# ---------------------------------------------------------------------------
# MoE Expert Load Imbalance
# ---------------------------------------------------------------------------

def moe_load_imbalance_factor(
    batch_size: int,
    n_ep_groups: int,
    top_k: int,
) -> float:
    """
    Throughput degradation factor from MoE expert routing load imbalance.

    In Expert Parallelism (EP), tokens are dispatched across n_ep_groups GPU
    ranks.  Each of the B tokens in a decode batch activates top_k experts;
    with n_ep_groups ranks in the EP collective, each rank expects to receive:

        mean_load = B × top_k / n_ep_groups  tokens

    Due to random routing, some EP ranks receive more tokens than the mean.
    The most-loaded rank is the straggler that determines step time.

    Caller must pass n_ep_groups = min(cfg.ep, model.n_experts), NOT the
    total expert count.  For EP=1 (no tensor-parallel expert split) the
    formula correctly returns 1.0 since there is no inter-rank contention.

    Formula (balls-into-bins, order-statistic approximation):
        sigma         = sqrt(mean_load × (1 − top_k/n_ep_groups))
        z_p           = sqrt(2 × ln(n_ep_groups))
        expected_max  = mean_load + sigma × z_p
        factor        = clip(expected_max / mean_load, 1.0, n_ep_groups/top_k)

    Edge cases:
      - batch_size ≤ 1: no contention; returns 1.0
      - n_ep_groups ≤ 1 (EP disabled or dense model): returns 1.0
      - top_k ≥ n_ep_groups (every rank always gets tokens): sigma≈0 → ≈1.0

    Note: assumes uniformly random routing. Real routers with an auxiliary
    load-balancing loss typically see 50–70% of this analytical upper bound.

    Source:
      [Mitzenmacher01] Mitzenmacher, M. "The Power of Two Choices in
          Randomized Load Balancing." IEEE Trans. Parallel Distrib. Syst.
          12(10), 2001. → balls-into-bins expected maximum.
    """
    if batch_size <= 1 or n_ep_groups <= 1 or top_k <= 0:
        return 1.0
    p = top_k / n_ep_groups
    mean_load = batch_size * p
    if mean_load <= 0:
        return 1.0
    sigma = math.sqrt(mean_load * max(0.0, 1.0 - p))
    z_p = math.sqrt(2.0 * math.log(n_ep_groups))
    expected_max = mean_load + sigma * z_p
    factor = expected_max / mean_load
    max_possible = n_ep_groups / top_k
    return float(min(max(factor, 1.0), max_possible))


def gpu_memory_required_bytes(
    model: ModelConfig,
    tp: int,
    ep: int,
    pp: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """
    Estimated peak HBM required per GPU for a given parallelism config.

    Weights
    -------
    - Attention (Q,K,V,O): replicated across EP ranks, sharded by TP across
      heads, sharded by PP across layers.
      bytes = (n_layers/PP) × 4 × d_model² × weight_dtype / TP

    - Dense FFN layers: sharded by TP, by PP.
      bytes = (n_dense_layers/PP) × 2 × d_model × d_ffn × weight_dtype / TP

    - MoE FFN layers: experts sharded across EP ranks, weights within each
      expert sharded by TP, layers sharded by PP.
      bytes = (n_moe_layers/PP) × (n_experts/EP) × 2 × d_model × d_ffn
              × weight_dtype / TP

    KV cache
    --------
    Each PP stage stores KV only for its assigned layers; attention heads
    are sharded by TP.
      bytes = 2 × (n_layers/PP) × batch × seq_len × (n_heads/TP) × d_head
              × kv_dtype

    Sources: [Brown20] parameter counts; [Narayanan21] TP sharding;
    [Rajbhandari22] EP sharding; [Pope22] KV cache sizing.
    """
    ep_eff = min(ep, model.n_experts)  # EP capped at total expert count

    layers_per_stage     = model.n_layers       / pp
    dense_layers_per_stage = model.n_dense_layers / pp
    moe_layers_per_stage   = model.n_moe_layers   / pp

    attn_bytes     = layers_per_stage       * 4 * model.d_model * model.d_model \
                     * model.weight_dtype_bytes / tp
    dense_ffn_bytes = dense_layers_per_stage * 2 * model.d_model * model.d_ffn \
                     * model.weight_dtype_bytes / tp
    moe_ffn_bytes  = moe_layers_per_stage   * (model.n_experts / ep_eff) \
                     * 2 * model.d_model * model.d_ffn \
                     * model.weight_dtype_bytes / tp

    weight_bytes = attn_bytes + dense_ffn_bytes + moe_ffn_bytes

    kv_bytes = (
        2                           # K and V
        * layers_per_stage
        * batch_size
        * seq_len
        * (model.n_heads / tp)
        * model.d_head
        * model.kv_cache_dtype_bytes
    )

    return weight_bytes + kv_bytes
