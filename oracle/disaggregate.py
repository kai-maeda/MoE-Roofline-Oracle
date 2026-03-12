"""
Disaggregated prefill/decode serving model.

In a disaggregated architecture, the inference cluster is split into two
specialised pools:
  - P-instances: dedicated prefill GPUs (process prompts → fill KV cache)
  - D-instances: dedicated decode GPUs  (generate tokens from KV cache)

This eliminates prefill-decode interference, making TTFT and TPOT independently
controllable. The KV cache is transferred over InfiniBand from P→D after each
prefill, which is the primary overhead of disaggregation.

Key analytical result: total GPU cost per token is *identical* to coupled
serving — disaggregation does not reduce GPU count, it only allows specialised
utilisation. The benefit is purely TTFT improvement and latency predictability.

=============================================================================
REFERENCES
=============================================================================

  [Distserve]  Zhong, Y. et al. "DistServe: Disaggregating Prefill and
               Decoding for Goodput-Optimized Large Language Model Serving."
               OSDI 2024.
               → P:D ratio derivation (Little's Law throughput matching);
                 continuous batching backpressure model.

  [Splitwise]  Patel, P. et al. "Splitwise: Efficient Generative LLM
               Inference Using Phase Splitting." ISCA 2024.
               → TTFT backpressure: avg wait ≈ decode_step × batch / 2.
               → Benefit scales with decode concurrency.

  [Mooncake]   Qin, R. et al. "Mooncake: A KVCache-centric Disaggregated
               Architecture for LLM Serving." arXiv:2407.00079, 2024.
               → KV transfer sizing; IB bandwidth as the binding constraint.

  [Pope22]     Pope, R. et al. "Efficiently Scaling Transformer Inference."
               MLSys 2023. Section 2.1 — KV cache byte formula.
=============================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple
import math

from oracle.hardware import GPU, NetworkFabric
from oracle.workload import ModelConfig, ModelStats
from oracle.roofline import roofline


# ---------------------------------------------------------------------------
# KV cache byte helpers
# ---------------------------------------------------------------------------

def kv_cache_bytes_per_request(model: ModelConfig, isl: int) -> float:
    """
    KV cache bytes for a single request after prefill completes.

        bytes = 2 (K+V) × n_layers × ISL × n_heads × d_head × kv_dtype_bytes

    Source: [Pope22] Section 2.1.

    NOTE: Assumes Multi-Head Attention (MHA). GQA/MQA/MLA models (e.g.
    DeepSeek-V2's Multi-head Latent Attention) have significantly smaller
    KV caches. This formula will overestimate KV transfer time for those
    models — treat as an upper bound.
    """
    return (
        2 * model.n_layers * isl * model.n_heads * model.d_head
        * model.kv_cache_dtype_bytes
    )


# ---------------------------------------------------------------------------
# DisaggregateResult
# ---------------------------------------------------------------------------

@dataclass
class DisaggregateResult:
    """
    Side-by-side comparison of coupled vs disaggregated serving for one
    (model, GPU, workload) combination.

    All timing fields are in seconds; _ms properties expose milliseconds
    for display.
    """
    model_name: str
    gpu_name: str

    isl: int    # Input sequence length
    osl: int    # Output sequence length

    # --- Raw timings (seconds) ---
    ttft_base_s: float       # Pure prefill compute (no queuing or transfer)
    decode_step_s: float     # One decode step at chosen batch size
    kv_transfer_s: float     # KV cache transfer time P→D over InfiniBand

    # Config
    decode_batch_size: int
    interconnect_bw_gbs: float   # GB/s used for KV transfer

    # --- Derived metrics ---

    @property
    def pd_ratio(self) -> float:
        """
        Optimal n_prefill_gpus / n_decode_gpus for throughput balance.

        Derivation (Little's Law):
          P-GPU throughput = 1 / ttft_base_s  req/s
          D-GPU throughput = 1 / (osl × decode_step_s) req/s
          Balance: n_p / ttft = n_d / (osl × decode_step_s)
          → pd_ratio = ttft_base_s / (osl × decode_step_s)

        pd_ratio < 1: decode-heavy (typical) — most GPUs are decode instances.
        pd_ratio > 1: prefill-heavy (long prompts, short outputs).

        Source: [Distserve] Section 3.1, throughput matching via Little's Law.
        """
        denom = self.osl * self.decode_step_s
        return self.ttft_base_s / denom if denom > 0 else float('inf')

    @property
    def ttft_coupled_s(self) -> float:
        """
        TTFT in coupled continuous-batching serving.

        In continuous batching a new prefill waits for in-flight decode steps.
        With decode batch B, the expected waiting time is half a decode step
        (uniform arrival within a step interval):

            waiting_time ≈ decode_step_s × B / 2

        Source: [Splitwise] Section 4, [Distserve] Section 3.2.
        """
        waiting = self.decode_step_s * self.decode_batch_size / 2.0
        return self.ttft_base_s + waiting

    @property
    def ttft_disagg_s(self) -> float:
        """
        TTFT in disaggregated serving.

        No decode backpressure. Only added latency is the KV cache transfer
        from P-GPU to D-GPU over InfiniBand:

            TTFT_disagg = ttft_base + kv_transfer_time

        Source: [Mooncake] Section 3; [Distserve] Section 4.
        """
        return self.ttft_base_s + self.kv_transfer_s

    @property
    def ttft_improvement_pct(self) -> float:
        """% reduction in TTFT going from coupled → disaggregated."""
        if self.ttft_coupled_s <= 0:
            return 0.0
        return (self.ttft_coupled_s - self.ttft_disagg_s) / self.ttft_coupled_s * 100.0

    @property
    def disagg_helps(self) -> bool:
        """
        Heuristic: disaggregation provides meaningful TTFT improvement when:
          1. KV transfer doesn't dominate (transfer < prefill time)
          2. Workload is decode-heavy (pd_ratio < 0.5)
          3. TTFT improvement exceeds 20%

        If kv_transfer_s ≥ ttft_base_s, disaggregation *hurts* TTFT.
        This occurs at very short ISL where prefill is fast but transfer
        overhead is bounded by network round-trip time.
        """
        if self.kv_transfer_s >= self.ttft_base_s:
            return False
        return self.pd_ratio < 0.5 and self.ttft_improvement_pct > 20.0

    @property
    def kv_bytes(self) -> float:
        """KV cache bytes transferred per request."""
        return self.kv_transfer_s * self.interconnect_bw_gbs * 1e9

    # Millisecond accessors for display
    @property
    def ttft_base_ms(self) -> float:
        return self.ttft_base_s * 1e3

    @property
    def ttft_coupled_ms(self) -> float:
        return self.ttft_coupled_s * 1e3

    @property
    def ttft_disagg_ms(self) -> float:
        return self.ttft_disagg_s * 1e3

    @property
    def decode_step_ms(self) -> float:
        return self.decode_step_s * 1e3

    @property
    def kv_transfer_ms(self) -> float:
        return self.kv_transfer_s * 1e3


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def compute_disaggregate(
    model: ModelConfig,
    gpu: GPU,
    fabric: NetworkFabric,
    isl: int,
    osl: int,
    decode_batch_size: int,
    dtype_compute_scale: float = 1.0,
) -> DisaggregateResult:
    """
    Compute disaggregated serving analysis for one (model, GPU, workload).

    Args:
        model:               Model architecture config
        gpu:                 Target hardware
        fabric:              Network fabric (provides inter_node_bw_gb for KV transfer)
        isl:                 Input sequence length (prompt tokens)
        osl:                 Output sequence length (generated tokens)
        decode_batch_size:   Concurrent decode requests per D-GPU
        dtype_compute_scale: FP8=2.0, BF16=1.0 (passed to roofline)
    """
    # Prefill: time to complete the full prompt pass
    prefill_stats = ModelStats(
        model=model, seq_len=isl,
        batch_size=1, phase="prefill",
    )
    prefill_result = roofline(
        gpu, prefill_stats.total_flops, prefill_stats.total_bytes,
        name=f"{model.name} prefill (S={isl})",
        dtype_compute_scale=dtype_compute_scale,
    )
    ttft_base_s = prefill_result.bottleneck_time_s

    # Decode: one forward-pass step at the given batch size
    decode_stats = ModelStats(
        model=model, seq_len=isl,
        batch_size=decode_batch_size, phase="decode",
    )
    decode_result = roofline(
        gpu, decode_stats.total_flops, decode_stats.total_bytes,
        name=f"{model.name} decode (ctx={isl}, B={decode_batch_size})",
        dtype_compute_scale=dtype_compute_scale,
    )
    decode_step_s = decode_result.bottleneck_time_s

    # KV transfer: P-GPU sends filled KV cache to D-GPU over IB
    kv_bytes = kv_cache_bytes_per_request(model, isl)
    kv_transfer_s = kv_bytes / (fabric.inter_node_bw_gb * 1e9) if fabric.inter_node_bw_gb > 0 else 0.0

    return DisaggregateResult(
        model_name=model.name,
        gpu_name=gpu.name,
        isl=isl,
        osl=osl,
        ttft_base_s=ttft_base_s,
        decode_step_s=decode_step_s,
        kv_transfer_s=kv_transfer_s,
        decode_batch_size=decode_batch_size,
        interconnect_bw_gbs=fabric.inter_node_bw_gb,
    )


# ---------------------------------------------------------------------------
# Heterogeneous constructor (separate GPU types for P and D instances)
# ---------------------------------------------------------------------------

def compute_disaggregate_hetero(
    model: ModelConfig,
    prefill_gpu: GPU,
    decode_gpu: GPU,
    fabric: NetworkFabric,
    isl: int,
    osl: int,
    decode_batch_size: int,
    dtype_compute_scale: float = 1.0,
) -> DisaggregateResult:
    """
    Heterogeneous disaggregated serving: separate GPU types for P and D pools.

    Prefill is compute-bound → use a compute-dense GPU (e.g. B200, H100 SXM5).
    Decode is memory-bandwidth-bound → use a bandwidth-dense GPU (e.g. NVL72, MI300X).

    The P:D ratio is derived from the two GPU throughputs via Little's Law:
        pd_ratio = ttft_base_s(prefill_gpu) / (osl × decode_step_s(decode_gpu))

    Args:
        model:               Model architecture config
        prefill_gpu:         GPU used for P-instances (prefill pool)
        decode_gpu:          GPU used for D-instances (decode pool)
        fabric:              Network fabric (IB bandwidth for KV transfer)
        isl:                 Input sequence length (prompt tokens)
        osl:                 Output sequence length (generated tokens)
        decode_batch_size:   Concurrent decode requests per D-GPU
        dtype_compute_scale: FP8=2.0, BF16=1.0
    """
    prefill_stats = ModelStats(model=model, seq_len=isl, batch_size=1, phase="prefill")
    prefill_result = roofline(
        prefill_gpu, prefill_stats.total_flops, prefill_stats.total_bytes,
        name=f"{model.name} prefill (S={isl})",
        dtype_compute_scale=dtype_compute_scale,
    )
    ttft_base_s = prefill_result.bottleneck_time_s

    decode_stats = ModelStats(model=model, seq_len=isl, batch_size=decode_batch_size, phase="decode")
    decode_result = roofline(
        decode_gpu, decode_stats.total_flops, decode_stats.total_bytes,
        name=f"{model.name} decode (ctx={isl}, B={decode_batch_size})",
        dtype_compute_scale=dtype_compute_scale,
    )
    decode_step_s = decode_result.bottleneck_time_s

    kv_bytes = kv_cache_bytes_per_request(model, isl)
    kv_transfer_s = kv_bytes / (fabric.inter_node_bw_gb * 1e9) if fabric.inter_node_bw_gb > 0 else 0.0

    return DisaggregateResult(
        model_name=model.name,
        gpu_name=f"{prefill_gpu.name} → {decode_gpu.name}",
        isl=isl,
        osl=osl,
        ttft_base_s=ttft_base_s,
        decode_step_s=decode_step_s,
        kv_transfer_s=kv_transfer_s,
        decode_batch_size=decode_batch_size,
        interconnect_bw_gbs=fabric.inter_node_bw_gb,
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep_disaggregate_batch(
    model: ModelConfig,
    gpu: GPU,
    fabric: NetworkFabric,
    isl: int,
    osl: int,
    batch_sizes: List[int],
    dtype_compute_scale: float = 1.0,
) -> List[DisaggregateResult]:
    """Sweep over decode batch sizes for a single (model, GPU, workload)."""
    return [
        compute_disaggregate(model, gpu, fabric, isl, osl, b, dtype_compute_scale)
        for b in batch_sizes
    ]


def sweep_disaggregate_workloads(
    model: ModelConfig,
    gpu: GPU,
    fabric: NetworkFabric,
    isl_osl_pairs: List[Tuple[int, int]],
    decode_batch_size: int,
    dtype_compute_scale: float = 1.0,
) -> List[DisaggregateResult]:
    """Sweep over (ISL, OSL) pairs for a single (GPU, batch) combination."""
    return [
        compute_disaggregate(model, gpu, fabric, isl, osl, decode_batch_size, dtype_compute_scale)
        for isl, osl in isl_osl_pairs
    ]
