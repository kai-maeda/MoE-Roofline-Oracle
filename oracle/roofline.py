"""
Roofline model engine.

The roofline model bounds achievable performance:

    P_achieved = min(π, β × I)                                         [WWP09, Eq. 4]

Where:
    π  = peak compute throughput (FLOP/s)  — from hardware datasheet
    β  = peak memory bandwidth (bytes/s)   — from hardware datasheet
    I  = arithmetic intensity (FLOPs/byte) — computed in workload.py

The ridge point I* = π / β (FLOPs/byte) separates the two regimes:
  - I < I*: memory-bandwidth bound   → P = β × I
  - I ≥ I*: compute bound            → P = π

For MoE inference:
  - Prefill is typically compute-bound (high I due to large batch / long sequence):
      I_prefill ~ 2*d_model / dtype_bytes  (weight-stationary regime)
  - Decode is almost always memory-bandwidth-bound (small batch, weight streaming):
      I_decode  ~ 2 / dtype_bytes          (one token, weights loaded cold)
  - Expert routing AlltoAll adds communication latency modeled in parallelism.py.

hardware_efficiency parameter accounts for:
  - Tensor core utilization gap (e.g., kernel launch overhead, small tiles)
  - Memory controller efficiency (bank conflicts, non-unit stride)
  - Empirically, production kernels achieve 85-90% of roofline peak for GEMMs.
  Source: [Dao22] FlashAttention benchmarks; [NVIDIA23] cuBLAS performance guide.

=============================================================================
CITATIONS
=============================================================================

  [WWP09]    Williams, S., Waterman, A., Patterson, D. "Roofline: An Insightful
             Visual Performance Model for Floating-Point Programs and Multiprocessors."
             Communications of the ACM 52(4), 2009. DOI:10.1145/1498765.1498785.
             → Core roofline formula P = min(π, β × I); ridge point I* = π/β.

  [Dao22]    Dao, T. et al. "FlashAttention." NeurIPS 2022.
             → Hardware efficiency benchmarks for attention kernels on A100/H100.

  [NVIDIA23] NVIDIA. "cuBLAS Library User Guide." (2023).
             → GEMM efficiency as fraction of peak TFLOP/s on Hopper/Ampere.

  [Pope22]   Pope, R. et al. "Efficiently Scaling Transformer Inference."
             MLSys 2023. Section 3 applies roofline to prefill vs. decode phases.
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from oracle.hardware import GPU, ClusterConfig
from oracle.workload import ModelConfig, ModelStats, OperatorStats


@dataclass
class RooflineResult:
    """Performance prediction for a single operator or full model pass."""
    name: str
    arithmetic_intensity: float    # FLOPs / byte
    peak_compute_tflops: float     # Hardware peak (TFLOP/s)
    peak_bandwidth_tbs: float      # Hardware peak bandwidth (TB/s)
    ridge_point: float             # I* = π/β (FLOPs/byte)

    # Achieved performance
    achieved_tflops: float         # min(π, β×I) before MFU penalty
    mfu_theoretical: float         # Model FLOPs utilization (0-1)

    # Bottleneck classification
    is_compute_bound: bool

    # Timing
    total_flops: float
    total_bytes: float
    compute_time_s: float          # time if purely compute bound
    memory_time_s: float           # time if purely memory bound
    bottleneck_time_s: float       # max(compute_time, memory_time)

    @property
    def is_memory_bound(self) -> bool:
        return not self.is_compute_bound

    @property
    def compute_time_ms(self) -> float:
        return self.compute_time_s * 1e3

    @property
    def memory_time_ms(self) -> float:
        return self.memory_time_s * 1e3

    @property
    def bottleneck_time_ms(self) -> float:
        return self.bottleneck_time_s * 1e3

    @property
    def bottleneck(self) -> str:
        return "compute-bound" if self.is_compute_bound else "memory-bound"


def roofline(
    gpu: GPU,
    flops: float,
    bytes_moved: float,
    name: str = "op",
    hardware_efficiency: Optional[float] = None,
    dtype_compute_scale: float = 1.0,
) -> RooflineResult:
    """
    Apply roofline model to a single operator.

    Args:
        gpu: Target hardware
        flops: Total floating point operations
        bytes_moved: Total bytes transferred to/from HBM
        name: Operator label
        hardware_efficiency: Override for both compute and BW efficiency (0-1).
            If None (default), uses gpu.sw_efficiency_compute and gpu.sw_efficiency_bw,
            which are per-GPU estimates based on community benchmarks.
        dtype_compute_scale: Multiplier on peak compute for dtype.
            FP8 = 2.0 (tensor cores process 2x ops vs BF16). BF16/FP16 = 1.0.
    """
    if hardware_efficiency is not None:
        eff_compute = hardware_efficiency
        eff_bw = hardware_efficiency
    else:
        eff_compute = gpu.sw_efficiency_compute
        eff_bw = gpu.sw_efficiency_bw

    peak_compute = gpu.flops_bf16 * 1e12 * eff_compute * dtype_compute_scale  # FLOP/s
    peak_bw = gpu.hbm_bandwidth_tb * 1e12 * eff_bw   # bytes/s
    ridge_point = peak_compute / peak_bw                           # FLOPs/byte

    intensity = flops / bytes_moved if bytes_moved > 0 else float('inf')

    # Roofline bound
    achieved = min(peak_compute, peak_bw * intensity)   # FLOP/s
    is_compute_bound = intensity >= ridge_point

    # Time estimates
    compute_time = flops / peak_compute if peak_compute > 0 else 0.0
    memory_time = bytes_moved / peak_bw if peak_bw > 0 else 0.0
    bottleneck_time = max(compute_time, memory_time)

    mfu = achieved / (gpu.flops_bf16 * 1e12) if gpu.flops_bf16 > 0 else 0.0

    return RooflineResult(
        name=name,
        arithmetic_intensity=intensity,
        peak_compute_tflops=gpu.flops_bf16,
        peak_bandwidth_tbs=gpu.hbm_bandwidth_tb,
        ridge_point=ridge_point,
        achieved_tflops=achieved / 1e12,
        mfu_theoretical=mfu,
        is_compute_bound=is_compute_bound,
        total_flops=flops,
        total_bytes=bytes_moved,
        compute_time_s=compute_time,
        memory_time_s=memory_time,
        bottleneck_time_s=bottleneck_time,
    )


@dataclass
class InferenceProfile:
    """
    Complete inference performance profile for a model on hardware.

    Covers prefill and decode phases separately, since they have
    fundamentally different bottleneck profiles.
    """
    model: ModelConfig
    gpu: GPU

    # Prefill
    prefill_seq_len: int
    prefill_batch_size: int

    # Decode
    decode_seq_len: int          # Context length being attended over
    decode_batch_size: int

    # Parallelism penalties applied externally
    parallelism_efficiency: float = 1.0  # 0-1, from parallelism.py

    # Dtype: multiplier on peak compute (FP8=2.0, BF16/FP16=1.0)
    dtype_compute_scale: float = 1.0

    # Computed in __post_init__
    prefill_result: RooflineResult = field(default=None, init=False)
    decode_result: RooflineResult = field(default=None, init=False)

    def __post_init__(self):
        self._run()

    def _run(self):
        # Prefill
        prefill_stats = ModelStats(
            model=self.model,
            seq_len=self.prefill_seq_len,
            batch_size=self.prefill_batch_size,
            phase="prefill",
        )
        self.prefill_result = roofline(
            gpu=self.gpu,
            flops=prefill_stats.total_flops,
            bytes_moved=prefill_stats.total_bytes,
            name=f"{self.model.name} prefill (S={self.prefill_seq_len}, B={self.prefill_batch_size})",
            dtype_compute_scale=self.dtype_compute_scale,
        )

        # Decode (single token, attending over full context)
        decode_stats = ModelStats(
            model=self.model,
            seq_len=self.decode_seq_len,
            batch_size=self.decode_batch_size,
            phase="decode",
        )
        self.decode_result = roofline(
            gpu=self.gpu,
            flops=decode_stats.total_flops,
            bytes_moved=decode_stats.total_bytes,
            name=f"{self.model.name} decode (ctx={self.decode_seq_len}, B={self.decode_batch_size})",
            dtype_compute_scale=self.dtype_compute_scale,
        )

    @property
    def tokens_per_second_prefill(self) -> float:
        """Throughput during prefill (tokens/s/GPU)."""
        t = self.prefill_result.bottleneck_time_s / self.parallelism_efficiency
        tokens = self.prefill_seq_len * self.prefill_batch_size
        return tokens / t if t > 0 else 0.0

    @property
    def tokens_per_second_decode(self) -> float:
        """Throughput during decode (tokens/s/GPU)."""
        t = self.decode_result.bottleneck_time_s / self.parallelism_efficiency
        tokens = self.decode_batch_size
        return tokens / t if t > 0 else 0.0

    @property
    def time_to_first_token_ms(self) -> float:
        """TTFT: time to complete prefill for one request."""
        t = self.prefill_result.bottleneck_time_s / self.parallelism_efficiency
        return t * 1e3

    @property
    def inter_token_latency_ms(self) -> float:
        """ITL: time per generated token in decode (single request, batch=1)."""
        decode_single = ModelStats(
            model=self.model,
            seq_len=self.decode_seq_len,
            batch_size=1,
            phase="decode",
        )
        r = roofline(self.gpu, decode_single.total_flops, decode_single.total_bytes,
                     dtype_compute_scale=self.dtype_compute_scale)
        t = r.bottleneck_time_s / self.parallelism_efficiency
        return t * 1e3

    def summary(self) -> dict:
        return {
            "model": self.model.name,
            "gpu": self.gpu.name,
            "prefill_bottleneck": self.prefill_result.bottleneck,
            "prefill_mfu": f"{self.prefill_result.mfu_theoretical:.1%}",
            "prefill_intensity": f"{self.prefill_result.arithmetic_intensity:.1f} FLOPs/byte",
            "decode_bottleneck": self.decode_result.bottleneck,
            "decode_mfu": f"{self.decode_result.mfu_theoretical:.1%}",
            "decode_intensity": f"{self.decode_result.arithmetic_intensity:.1f} FLOPs/byte",
            "ttft_ms": f"{self.time_to_first_token_ms:.1f}",
            "itl_ms": f"{self.inter_token_latency_ms:.2f}",
            "tok_s_decode": f"{self.tokens_per_second_decode:.0f}",
        }


def sweep_batch_sizes(
    model: ModelConfig,
    gpu: GPU,
    seq_len: int,
    batch_sizes: list,
    phase: str = "decode",
) -> list:
    """
    Sweep arithmetic intensity and MFU across batch sizes.
    Returns list of (batch_size, RooflineResult).
    """
    results = []
    for bs in batch_sizes:
        stats = ModelStats(model=model, seq_len=seq_len, batch_size=bs, phase=phase)
        r = roofline(gpu, stats.total_flops, stats.total_bytes,
                     name=f"B={bs}")
        results.append((bs, r))
    return results


def sweep_sequence_lengths(
    model: ModelConfig,
    gpu: GPU,
    seq_lens: list,
    batch_size: int = 1,
    phase: str = "prefill",
) -> list:
    """Sweep arithmetic intensity across sequence lengths."""
    results = []
    for sl in seq_lens:
        stats = ModelStats(model=model, seq_len=sl, batch_size=batch_size, phase=phase)
        r = roofline(gpu, stats.total_flops, stats.total_bytes,
                     name=f"S={sl}")
        results.append((sl, r))
    return results
