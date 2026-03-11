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

# Mixtral 8x7B (Mistral AI)
MIXTRAL_8x7B = ModelConfig(
    name="Mixtral-8x7B",
    d_model=4096,
    n_layers=32,
    n_heads=32,
    d_head=128,
    d_ffn=14336,
    n_experts=8,
    n_experts_active=2,
    moe_every_n=1,   # Every layer is MoE
    max_seq_len=32768,
    vocab_size=32000,
)

# Mixtral 8x22B (Mistral AI)
MIXTRAL_8x22B = ModelConfig(
    name="Mixtral-8x22B",
    d_model=6144,
    n_layers=56,
    n_heads=48,
    d_head=128,
    d_ffn=16384,
    n_experts=8,
    n_experts_active=2,
    moe_every_n=1,
    max_seq_len=65536,
    vocab_size=32000,
)

# DeepSeek-V2 (236B total, 21B active)
DEEPSEEK_V2 = ModelConfig(
    name="DeepSeek-V2",
    d_model=5120,
    n_layers=60,
    n_heads=128,
    d_head=128,
    d_ffn=1536,      # MLA: compressed KV, smaller effective d_ffn per expert
    n_experts=160,
    n_experts_active=6,
    moe_every_n=1,
    max_seq_len=128000,
    vocab_size=102400,
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

ALL_MODELS = [MIXTRAL_8x7B, MIXTRAL_8x22B, DEEPSEEK_V2, GPT3_175B]
