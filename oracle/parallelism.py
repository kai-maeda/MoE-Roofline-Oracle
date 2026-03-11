"""
Parallelism overhead modeling for distributed MoE inference.

Three dominant strategies:
  - Tensor Parallelism (TP): splits weight matrices across GPUs within a node.
    Requires AllReduce after each GEMM → high BW, low latency demand.
  - Pipeline Parallelism (PP): splits layers across nodes.
    Introduces pipeline bubble overhead → reduced throughput.
  - Expert Parallelism (EP): distributes experts across GPUs.
    Requires AlltoAll for token routing → the dominant cost for MoE at scale.

=============================================================================
CITATIONS
=============================================================================

Ring AllReduce communication time
  [Rabenseifner04] Rabenseifner, R. "Optimization of Collective Reduction
                   Operations." ICCS 2004. Derives ring-allreduce time:
                       t = 2*(N-1)/N * M / B + 2*(N-1) * α
                   where M = message size, B = bandwidth, α = latency,
                   N = number of ranks. We use a simplified version:
                       t_AR ≈ 2*(N-1)/N * M / B + α
                   Source for formula used in TP overhead below.

Tensor Parallelism communication pattern
  [Narayanan21]  Narayanan, D. et al. "Efficient Large-Scale Language Model
                 Training on GPU Clusters Using Megatron-LM." SC'21.
                 Section 3: TP requires AllReduce of (B, S, d_model) tensor
                 after attention output projection and after FFN down-projection
                 (2 AllReduces per transformer layer). Applied here to inference
                 forward pass only (backward pass AllReduces omitted).

Expert Parallelism (AlltoAll) communication pattern
  [Lepikhin21]   Lepikhin, D. et al. "GShard." ICLR 2021.
                 Section 2.3: AlltoAll dispatch sends each token to its assigned
                 expert GPU. AlltoAll gather returns expert outputs to origin GPU.
                 2 AlltoAlls per MoE layer (dispatch + gather).

  [Rajbhandari22] Rajbhandari, S. et al. "DeepSpeed-MoE." ICML 2022.
                  Section 4.2: Expert parallel AlltoAll message size =
                  (B * S * k / EP) * d_model * dtype_bytes per rank.

Pipeline bubble fraction
  [Huang19]      Huang, Y. et al. "GPipe: Efficient Training of Giant Neural
                 Networks using Pipeline Parallelism." NeurIPS 2019.
                 Section 3: pipeline bubble fraction = (PP-1) / (PP-1 + M)
                 where M = number of microbatches. For large M, bubble → 0.
                 Applied here to inference throughput (not training).

Amdahl's Law application to parallelism efficiency
  [Amdahl67]     Amdahl, G. "Validity of the Single Processor Approach to
                 Achieving Large Scale Computing Capabilities." AFIPS 1967.
                 Speedup = 1 / (f_serial + (1-f_serial)/N).
                 Here we express as efficiency = compute_time / total_time,
                 where total_time = compute_time + communication_time.
=============================================================================
"""

from dataclasses import dataclass
import numpy as np

from oracle.hardware import GPU, NetworkFabric, ClusterConfig


@dataclass
class ParallelismConfig:
    tp: int = 1     # Tensor parallel degree
    pp: int = 1     # Pipeline parallel degree
    ep: int = 1     # Expert parallel degree (for MoE)
    dp: int = 1     # Data parallel degree

    def __post_init__(self):
        assert self.tp >= 1
        assert self.pp >= 1
        assert self.ep >= 1

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.ep * self.dp

    def __str__(self):
        return f"TP={self.tp} PP={self.pp} EP={self.ep} DP={self.dp}"


@dataclass
class CollectiveStats:
    """Timing for a single collective communication operation."""
    name: str
    message_size_bytes: float    # Per-GPU message size
    algorithm: str               # "allreduce", "alltoall", "allgather", "reducescatter"
    num_ranks: int

    intra_node_bw_gbs: float     # Effective intra-node BW (NVLink/IF)
    inter_node_bw_gbs: float     # Effective inter-node BW (IB/RoCE)
    latency_us: float            # Baseline latency

    @property
    def allreduce_time_s(self) -> float:
        """
        Ring-AllReduce time.

        Formula: t = 2*(N-1)/N * M/B + α
        Source: [Rabenseifner04], simplified (latency term collapsed to α).
        """
        bw = self._effective_bw_bs()
        alpha = self.latency_us * 1e-6
        beta = 1.0 / bw if bw > 0 else float('inf')
        n = self.num_ranks
        return 2 * (n - 1) / n * self.message_size_bytes * beta + alpha

    @property
    def alltoall_time_s(self) -> float:
        """AlltoAll: each rank sends (N-1)/N of message to others."""
        bw = self._effective_bw_bs()
        alpha = self.latency_us * 1e-6
        n = self.num_ranks
        # Each GPU sends M*(N-1)/N bytes total
        send_bytes = self.message_size_bytes * (n - 1) / n
        return send_bytes / bw + alpha if bw > 0 else float('inf')

    @property
    def allgather_time_s(self) -> float:
        """AllGather: broadcast shard to all ranks."""
        bw = self._effective_bw_bs()
        alpha = self.latency_us * 1e-6
        n = self.num_ranks
        return (n - 1) / n * self.message_size_bytes * (1.0 / bw) + alpha if bw > 0 else float('inf')

    def _effective_bw_bs(self) -> float:
        """GB/s → bytes/s, using inter-node BW for large collectives."""
        return self.inter_node_bw_gbs * 1e9


def tp_allreduce_overhead(
    model_d: int,
    batch_size: int,
    seq_len: int,
    tp: int,
    fabric: NetworkFabric,
    dtype_bytes: float = 2.0,
    phase: str = "prefill",
) -> float:
    """
    Time for AllReduce after each TP-sharded GEMM.

    In tensor parallelism, each attention/FFN GEMM is split across TP GPUs.
    After each GEMM, we need an AllReduce of shape (B, S, d_model).
    Called twice per layer (attention output + FFN output).

    Returns: time in seconds per layer
    """
    S = seq_len if phase == "prefill" else 1
    # Message size: activation tensor (B, S, d_model)
    msg_bytes = batch_size * S * model_d * dtype_bytes

    # Use NVLink for intra-node TP (TP <= 8 typically fits on one node)
    bw_gbs = fabric.intra_node_bw_gb if tp <= 8 else fabric.inter_node_bw_gb
    bw_bs = bw_gbs * 1e9

    alpha = fabric.latency_us * 1e-6
    # Ring AllReduce: t = 2*(tp-1)/tp * M/B + α
    # Source: [Rabenseifner04]; [Narayanan21] Section 3 for TP application.
    allreduce_time = 2 * (tp - 1) / tp * msg_bytes / bw_bs + alpha

    # 2 AllReduces per layer (after attention and FFN)
    return 2 * allreduce_time


def ep_alltoall_overhead(
    model_d: int,
    batch_size: int,
    seq_len: int,
    ep: int,
    n_experts_active: int,
    fabric: NetworkFabric,
    dtype_bytes: float = 2.0,
    phase: str = "prefill",
) -> float:
    """
    Time for AlltoAll token dispatch + gather in Expert Parallelism.

    In EP, tokens are routed to experts on different GPUs via AlltoAll.
    This happens twice per MoE layer: dispatch (tokens → experts) and
    gather (expert outputs → tokens).

    Returns: time in seconds per MoE layer
    """
    S = seq_len if phase == "prefill" else 1
    # Tokens dispatched: each of B*S tokens goes to k experts
    # With EP, tokens scatter across ep GPUs
    tokens_per_gpu = batch_size * S * n_experts_active / ep
    msg_bytes = tokens_per_gpu * model_d * dtype_bytes

    # EP typically spans multiple nodes → use inter-node BW
    bw_gbs = fabric.inter_node_bw_gb if ep > 8 else fabric.intra_node_bw_gb
    bw_bs = bw_gbs * 1e9

    alpha = fabric.latency_us * 1e-6
    # AlltoAll: t = M*(ep-1)/ep / B + α
    # Each GPU sends its share of tokens to ep-1 other GPUs.
    # Source: [Rajbhandari22] Section 4.2; [Lepikhin21] Section 2.3.
    alltoall_time = msg_bytes * (ep - 1) / ep / bw_bs + alpha

    # 2 AlltoAlls per MoE layer (dispatch tokens → experts, gather outputs → tokens)
    # Source: [Lepikhin21] Section 2.3.
    return 2 * alltoall_time


def pp_bubble_fraction(pp: int, n_microbatches: int) -> float:
    """
    Pipeline bubble: fraction of time wasted in pipeline startup/drain.

    Formula: bubble = (PP - 1) / (PP - 1 + M)
    where PP = pipeline stages, M = number of microbatches.

    Derivation: the pipeline must fill (PP-1) stages before steady state and
    drain (PP-1) stages at the end. Total idle stages = 2*(PP-1) half-steps,
    which over (PP-1+M) steps gives the fraction above.

    Source: [Huang19] GPipe, Section 3 (Eq. 2); also [Narayanan21] Section 4.
    Note: Applied here to inference throughput, not training. In practice,
    continuous batching schemes (e.g., vLLM) reduce the bubble further.

    Returns fraction of time wasted (0 = no waste, 1 = all waste).
    """
    if pp == 1:
        return 0.0
    return (pp - 1) / (pp - 1 + n_microbatches)


@dataclass
class ParallelismOverhead:
    """
    Total communication overhead for a full model forward pass.
    All times in seconds.
    """
    config: ParallelismConfig

    # Per-layer times
    tp_time_per_layer_s: float = 0.0
    ep_time_per_moe_layer_s: float = 0.0

    # Global overheads
    pp_bubble_fraction: float = 0.0

    n_layers: int = 1
    n_moe_layers: int = 0

    @property
    def total_tp_time_s(self) -> float:
        return self.tp_time_per_layer_s * self.n_layers

    @property
    def total_ep_time_s(self) -> float:
        return self.ep_time_per_moe_layer_s * self.n_moe_layers

    @property
    def total_comm_time_s(self) -> float:
        return self.total_tp_time_s + self.total_ep_time_s

    def effective_efficiency(self, compute_time_s: float) -> float:
        """
        Fraction of time spent on useful compute vs total (compute + communication).

        Applies Amdahl's Law [Amdahl67]:
            efficiency = t_compute / (t_compute + t_comm) × (1 - bubble_fraction)

        The communication term is the "serial" fraction that cannot be parallelized
        (assuming compute and communication are not overlapped). Overlapping via
        double-buffering or async collectives would reduce this penalty in practice;
        this formulation is a conservative upper bound on overhead.

        Source: [Amdahl67]; [Narayanan21] Section 5 for practical MFU discussion.
        """
        bubble_penalty = 1.0 - self.pp_bubble_fraction
        total_time = compute_time_s + self.total_comm_time_s
        if total_time == 0:
            return 1.0
        compute_fraction = compute_time_s / total_time
        return compute_fraction * bubble_penalty


def compute_parallelism_overhead(
    config: ParallelismConfig,
    model_d: int,
    n_layers: int,
    n_moe_layers: int,
    n_experts_active: int,
    batch_size: int,
    seq_len: int,
    fabric: NetworkFabric,
    phase: str = "decode",
    n_microbatches: int = 8,
    dtype_bytes: float = 2.0,
) -> ParallelismOverhead:
    """Compute all parallelism communication overheads for a forward pass."""

    tp_overhead = 0.0
    if config.tp > 1:
        tp_overhead = tp_allreduce_overhead(
            model_d=model_d,
            batch_size=batch_size,
            seq_len=seq_len,
            tp=config.tp,
            fabric=fabric,
            dtype_bytes=dtype_bytes,
            phase=phase,
        )

    ep_overhead = 0.0
    if config.ep > 1 and n_moe_layers > 0:
        ep_overhead = ep_alltoall_overhead(
            model_d=model_d,
            batch_size=batch_size,
            seq_len=seq_len,
            ep=config.ep,
            n_experts_active=n_experts_active,
            fabric=fabric,
            dtype_bytes=dtype_bytes,
            phase=phase,
        )

    bubble = pp_bubble_fraction(config.pp, n_microbatches)

    return ParallelismOverhead(
        config=config,
        tp_time_per_layer_s=tp_overhead,
        ep_time_per_moe_layer_s=ep_overhead,
        pp_bubble_fraction=bubble,
        n_layers=n_layers,
        n_moe_layers=n_moe_layers,
    )


def sweep_parallelism_configs(
    model_d: int,
    n_layers: int,
    n_moe_layers: int,
    n_experts_active: int,
    batch_size: int,
    seq_len: int,
    fabric: NetworkFabric,
    compute_time_s: float,
    phase: str = "decode",
    configs: list = None,
) -> list:
    """
    Sweep over parallelism configs and return (config, overhead, efficiency).

    Useful for finding the Pareto-optimal parallelism strategy.
    """
    if configs is None:
        configs = [
            ParallelismConfig(tp=1, pp=1, ep=1),
            ParallelismConfig(tp=2, pp=1, ep=1),
            ParallelismConfig(tp=4, pp=1, ep=1),
            ParallelismConfig(tp=8, pp=1, ep=1),
            ParallelismConfig(tp=1, pp=1, ep=4),
            ParallelismConfig(tp=1, pp=1, ep=8),
            ParallelismConfig(tp=2, pp=1, ep=4),
            ParallelismConfig(tp=2, pp=2, ep=4),
            ParallelismConfig(tp=4, pp=2, ep=8),
        ]

    results = []
    for cfg in configs:
        oh = compute_parallelism_overhead(
            config=cfg,
            model_d=model_d,
            n_layers=n_layers,
            n_moe_layers=n_moe_layers,
            n_experts_active=n_experts_active,
            batch_size=batch_size,
            seq_len=seq_len,
            fabric=fabric,
            phase=phase,
        )
        eff = oh.effective_efficiency(compute_time_s)
        results.append((cfg, oh, eff))

    return results
