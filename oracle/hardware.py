"""
Hardware catalog: GPU specs, interconnect topology, and BoM cost estimates.

All bandwidth figures are unidirectional peak unless noted.
FLOPs are for BF16/FP16 dense GEMM (tensor core peak, no sparsity).
Cost figures are approximate street/list prices as of early 2025.

=============================================================================
SPEC CITATIONS
=============================================================================

NVIDIA H100 SXM5
  [H100-WP]  NVIDIA H100 Tensor Core GPU Architecture Whitepaper, NVIDIA (2022).
             https://resources.nvidia.com/en-us-tensor-core
  - BF16 TFLOPS: 989 (dense, tensor core) — Table 1, [H100-WP]
  - HBM3 bandwidth: 3.35 TB/s — Table 1, [H100-WP]
  - HBM3 capacity: 80 GB — Table 1, [H100-WP]
  - NVLink 4 bandwidth: 900 GB/s bidirectional (450 GB/s unidirectional) — [H100-WP]
  - TDP: 700 W — [H100-WP]
  - Price: ~$30,000 USD street (2024). Reported in NVIDIA partner pricing sheets;
    corroborated by SemiAnalysis HPC GPU pricing tracker (2024).

NVIDIA H200 SXM
  [H200-PB]  NVIDIA H200 Tensor Core GPU Product Brief, NVIDIA (2023).
             https://www.nvidia.com/en-us/data-center/h200/
  - BF16 TFLOPS: 989 (same Hopper compute die as H100) — [H200-PB]
  - HBM3e bandwidth: 4.8 TB/s — [H200-PB]
  - HBM3e capacity: 141 GB — [H200-PB]
  - NVLink 4 bandwidth: 900 GB/s bidirectional — [H200-PB]
  - TDP: ~700 W — [H200-PB]
  - Price: ~$40,000 USD estimated (H200 premium over H100).

NVIDIA B200 SXM
  [B200-WP]  NVIDIA Blackwell Architecture Technical Brief, NVIDIA (2024).
             https://www.nvidia.com/en-us/data-center/b200/
  - BF16 TFLOPS: 2,250 (dense tensor core) — [B200-WP]
  - HBM3e bandwidth: 8.0 TB/s — [B200-WP]
  - HBM3e capacity: 192 GB — [B200-WP]
  - NVLink 5 bandwidth: 1,800 GB/s bidirectional (900 GB/s unidirectional) — [B200-WP]
  - TDP: ~1,000 W — [B200-WP]
  - Price: ~$70,000 USD estimated (based on reported rack pricing; individual GPU
    street pricing not publicly listed as of early 2025).

NVIDIA B300 SXM  [ESTIMATED — post-announcement specs, subject to revision]
  [B300-ANN] NVIDIA Blackwell Ultra GTC 2025 Announcement. NVIDIA (2025).
  NOTE: B300 ("Blackwell Ultra") specs are based on NVIDIA's announced roadmap.
  Not all figures are confirmed in a published datasheet as of this writing.
  - BF16 TFLOPS: ~2,500 (estimated ~10-15% uplift over B200) — [B300-ANN]
  - HBM3e bandwidth: ~9.0 TB/s — [B300-ANN]
  - HBM3e capacity: 288 GB — [B300-ANN] (reported in early product briefs)
  - NVLink 5 bandwidth: 1,800 GB/s bidirectional — [B300-ANN]
  - TDP: ~1,200 W estimated
  - Price: ~$90,000 USD estimated

NVIDIA GB200 NVL72
  [NVL72-WP] NVIDIA GB200 NVL72 and NVL36 Datasheet, NVIDIA (2024).
             https://www.nvidia.com/en-us/data-center/gb200-nvl72/
  NOTE: The GB200 NVL72 is a rack-scale system of 72 B200 GPUs connected via a
  full NVLink 5 switch fabric (NVLink Switch chips). Per-GPU specs are identical
  to B200 SXM; the key difference is the nvlink_domain (72 vs 8).
  - Per-GPU BF16 TFLOPS: 2,250 — [NVL72-WP]
  - Per-GPU HBM3e bandwidth: 8.0 TB/s — [NVL72-WP]
  - Per-GPU HBM3e capacity: 192 GB — [NVL72-WP]
  - NVLink 5 per-GPU bandwidth: 1,800 GB/s bidirectional — [NVL72-WP]
  - NVLink domain: 72 GPUs (full rack, no IB required for intra-rack comms) — [NVL72-WP]
  - TDP (system): ~600 kW total rack — [NVL72-WP]; ~1,000 W/GPU imputed
  - Rack price: ~$3M USD estimated (NVIDIA partner conversations, 2024).
    Per-GPU equivalent: ~$41,700 USD.

NVIDIA GB300 NVL72  [ESTIMATED — post-announcement specs, subject to revision]
  [GB300-ANN] NVIDIA Blackwell Ultra GB300 NVL72 Announcement. NVIDIA (2025).
  NOTE: GB300 NVL72 replaces B200 GPUs with B300 GPUs in the same rack form
  factor. Specs extrapolated from B300 SXM and GB200 NVL72 architecture.
  - Per-GPU BF16 TFLOPS: ~2,500 — [GB300-ANN]
  - Per-GPU HBM3e bandwidth: ~9.0 TB/s — [GB300-ANN]
  - Per-GPU HBM3e capacity: 288 GB — [GB300-ANN]
  - NVLink 5 per-GPU bandwidth: 1,800 GB/s bidirectional — [GB300-ANN]
  - NVLink domain: 72 GPUs
  - Rack price: ~$4M USD estimated.

AMD MI300X
  [MI300X-PB] AMD Instinct MI300X Accelerator Product Brief, AMD (2024).
              https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
  - BF16 TFLOPS: 1,307 — [MI300X-PB], Table: Peak Throughput
  - HBM3 bandwidth: 5.3 TB/s — [MI300X-PB]
  - HBM3 capacity: 192 GB — [MI300X-PB]
  - AMD Infinity Fabric: 896 GB/s bidirectional — [MI300X-PB]
  - TDP: 750 W — [MI300X-PB]
  - Price: ~$15,000 USD (aggressive OEM pricing, 2024). Corroborated by
    cloud instance pricing reverse-engineering by SemiAnalysis (2024).

AMD MI325X
  [MI325X-PB] AMD Instinct MI325X Accelerator Product Brief, AMD (2024).
              https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
  - BF16 TFLOPS: 1,307 (same CDNA3 compute die as MI300X) — [MI325X-PB]
  - HBM3e bandwidth: 6.0 TB/s (vs 5.3 TB/s on MI300X) — [MI325X-PB]
  - HBM3e capacity: 256 GB (vs 192 GB on MI300X) — [MI325X-PB]
  - AMD Infinity Fabric: 896 GB/s bidirectional — [MI325X-PB]
  - TDP: ~750 W — [MI325X-PB]
  - Price: ~$20,000 USD estimated

AMD MI355X
  [MI355X-PB] AMD Instinct MI355X (CDNA4) Product Brief, AMD (2025).
              https://www.amd.com/en/products/accelerators/instinct/mi300/mi355x.html
  NOTE: MI355X uses AMD CDNA4 architecture — a generational jump from CDNA3.
  - BF16 TFLOPS: 2,560 (CDNA4, dense) — [MI355X-PB]
  - HBM3e bandwidth: 8.0 TB/s — [MI355X-PB]
  - HBM3e capacity: 288 GB — [MI355X-PB]
  - AMD Infinity Fabric: 1,792 GB/s bidirectional (CDNA4 upgrade) — [MI355X-PB]
  - TDP: ~1,000 W estimated
  - Price: ~$35,000 USD estimated

Network Fabrics
  [IB-HDR]   NVIDIA InfiniBand HDR (200Gb/s) Switch Datasheet, NVIDIA/Mellanox (2020).
  [IB-NDR]   NVIDIA InfiniBand NDR (400Gb/s) Switch Datasheet, NVIDIA/Mellanox (2022).
  - HDR per-port bandwidth: 200 Gb/s = 25 GB/s
  - NDR per-port bandwidth: 400 Gb/s = 50 GB/s
  - Switch costs: estimated from enterprise list pricing (2024).

=============================================================================
ECONOMIC ASSUMPTIONS — see cost.py for full methodology
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPU:
    name: str

    # Compute
    flops_bf16: float          # Peak BF16 TFLOP/s (tensor cores, dense, no sparsity)
    flops_fp32: float          # Peak FP32 TFLOP/s

    # Memory
    hbm_bandwidth_tb: float    # HBM bandwidth TB/s (unidirectional peak)
    hbm_capacity_gb: float     # HBM capacity GB

    # On-node interconnect (NVLink / Infinity Fabric)
    nvlink_bw_gb: float        # Unidirectional NVLink/IF bandwidth GB/s per GPU
    nvlink_domain: int         # Max GPUs in one NVLink domain (1 node or full rack)

    # Power
    tdp_w: float               # TDP watts

    # Economics — hardware purchase
    gpu_cost_usd: float        # Approximate unit cost USD
    server_cost_usd: float     # Full server chassis cost (excl. GPUs)

    # Economics — hardware-specific lifecycle defaults
    # These are baked in per-GPU because they reflect market dynamics and
    # ecosystem maturity, not operator choice (unlike power cost / PUE).
    depreciation_years: float = 3.0       # Expected refresh cycle (years)
    typical_utilization: float = 0.45     # Achievable cluster utilization
    opex_overhead_fraction: float = 0.10  # Non-power OpEx as fraction of CapEx/yr

    # Metadata
    architecture: str = ""     # e.g. "Hopper", "Blackwell", "CDNA3"
    specs_confirmed: bool = True  # False = estimated / pre-datasheet

    @property
    def hbm_bandwidth_gbs(self) -> float:
        return self.hbm_bandwidth_tb * 1e3

    @property
    def arithmetic_intensity_ridge(self) -> float:
        """
        Roofline ridge point: arithmetic intensity (FLOPs/byte) at which the
        operation transitions from memory-bound to compute-bound.

        I* = π / β,  where π = peak FLOP/s, β = peak memory BW (bytes/s)

        Source: Williams, Waterman, Patterson (2009), "Roofline: An Insightful
        Visual Performance Model for Floating-Point Programs and Multiprocessors",
        Comm. ACM 52(4). Equation (1).
        """
        return (self.flops_bf16 * 1e12) / (self.hbm_bandwidth_tb * 1e12)

    @property
    def cost_per_tflop(self) -> float:
        """USD per BF16 TFLOP/s. Lower = more compute-efficient purchase."""
        return self.gpu_cost_usd / self.flops_bf16

    @property
    def cost_per_tb_bandwidth(self) -> float:
        """USD per TB/s HBM bandwidth. Lower = more bandwidth-efficient purchase."""
        return self.gpu_cost_usd / self.hbm_bandwidth_tb

    @property
    def memory_capacity_per_dollar(self) -> float:
        """GB per USD. Relevant for model fitting (KV cache, weights)."""
        return self.hbm_capacity_gb / self.gpu_cost_usd


@dataclass
class NetworkFabric:
    name: str
    intra_node_bw_gb: float     # Intra-node (NVLink/IF) unidirectional GB/s
    inter_node_bw_gb: float     # Inter-node (IB/RoCE) unidirectional GB/s per GPU
    latency_us: float           # Baseline collective latency µs
    switch_cost_usd: float      # Per-port cost of top-of-rack switch USD
    ports_per_switch: int       # Ports per switch


@dataclass
class ClusterConfig:
    gpu: GPU
    fabric: NetworkFabric
    num_nodes: int
    gpus_per_node: int = 8

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @property
    def total_flops_bf16(self) -> float:
        """Aggregate cluster peak BF16 TFLOP/s."""
        return self.total_gpus * self.gpu.flops_bf16

    @property
    def capex_gpu_usd(self) -> float:
        return self.total_gpus * self.gpu.gpu_cost_usd

    @property
    def capex_server_usd(self) -> float:
        return self.num_nodes * self.gpu.server_cost_usd

    @property
    def capex_network_usd(self) -> float:
        """
        Networking CapEx estimate.

        Fat-tree topology: approximately 2 layers of switches for clusters up
        to ~128 nodes. We estimate 1 switch per 4 nodes (leaf layer) plus
        1 spine switch per 16 nodes, rounded up.

        Switch port costs from [IB-HDR] and [IB-NDR] enterprise pricing.
        """
        switches_needed = max(1, self.num_nodes // 4)
        return switches_needed * self.fabric.switch_cost_usd * self.fabric.ports_per_switch

    @property
    def total_capex_usd(self) -> float:
        return self.capex_gpu_usd + self.capex_server_usd + self.capex_network_usd

    @property
    def total_tdp_kw(self) -> float:
        return (self.total_gpus * self.gpu.tdp_w) / 1000.0


# =============================================================================
# Hardware Catalog
# =============================================================================
# All specs cited in module docstring above.
# specs_confirmed=False entries are based on announced roadmap specs and may
# differ from final shipping silicon.
# =============================================================================

# --- NVIDIA Hopper Generation ---

NVIDIA_H100_SXM = GPU(
    name="NVIDIA H100 SXM5",
    architecture="Hopper",
    flops_bf16=989.0,          # 989 TFLOP/s BF16 dense — [H100-WP]
    flops_fp32=67.0,           # 67 TFLOP/s FP32 — [H100-WP]
    hbm_bandwidth_tb=3.35,     # 3.35 TB/s HBM3 — [H100-WP]
    hbm_capacity_gb=80.0,      # 80 GB — [H100-WP]
    nvlink_bw_gb=450.0,        # 900 GB/s bidirectional → 450 GB/s unidirectional — [H100-WP]
    nvlink_domain=8,
    tdp_w=700.0,               # [H100-WP]
    gpu_cost_usd=30_000.0,     # ~$30k street (2024)
    server_cost_usd=120_000.0, # DGX H100 chassis (excl. GPUs) ~$120k
    depreciation_years=3.0,    # Mature generation; standard 3-yr cycle
    typical_utilization=0.50,  # CUDA ecosystem maturity → high utilization
    opex_overhead_fraction=0.10,
    specs_confirmed=True,
)

NVIDIA_H200_SXM = GPU(
    name="NVIDIA H200 SXM",
    architecture="Hopper",
    flops_bf16=989.0,          # Same Hopper compute die — [H200-PB]
    flops_fp32=67.0,
    hbm_bandwidth_tb=4.8,      # 4.8 TB/s HBM3e (+43% over H100) — [H200-PB]
    hbm_capacity_gb=141.0,     # 141 GB HBM3e — [H200-PB]
    nvlink_bw_gb=450.0,        # NVLink 4 same as H100 — [H200-PB]
    nvlink_domain=8,
    tdp_w=700.0,               # [H200-PB]
    gpu_cost_usd=40_000.0,     # ~$40k estimated (H200 premium)
    server_cost_usd=130_000.0,
    depreciation_years=3.0,    # Same Hopper die; same cycle as H100
    typical_utilization=0.48,  # Slightly newer → slightly lower adoption than H100
    opex_overhead_fraction=0.10,
    specs_confirmed=True,
)

# --- NVIDIA Blackwell Generation ---

NVIDIA_B200_SXM = GPU(
    name="NVIDIA B200 SXM",
    architecture="Blackwell",
    flops_bf16=2_250.0,        # 2,250 TFLOP/s BF16 dense — [B200-WP]
    flops_fp32=150.0,          # Estimated from die area scaling vs H100
    hbm_bandwidth_tb=8.0,      # 8.0 TB/s HBM3e — [B200-WP]
    hbm_capacity_gb=192.0,     # 192 GB — [B200-WP]
    nvlink_bw_gb=900.0,        # 1,800 GB/s bidirectional → 900 GB/s unidirectional — [B200-WP]
    nvlink_domain=8,
    tdp_w=1_000.0,             # ~1000 W — [B200-WP]
    gpu_cost_usd=70_000.0,     # ~$70k estimated (per-GPU equiv from rack pricing)
    server_cost_usd=200_000.0,
    depreciation_years=2.5,    # Premium price → faster refresh to stay competitive
    typical_utilization=0.38,  # Ramping deployment; ecosystem still maturing
    opex_overhead_fraction=0.11,
    specs_confirmed=True,
)

NVIDIA_B300_SXM = GPU(
    name="NVIDIA B300 SXM (Blackwell Ultra)",
    architecture="Blackwell Ultra",
    flops_bf16=2_500.0,        # ~2,500 TFLOP/s BF16 dense — [B300-ANN] (estimated)
    flops_fp32=170.0,          # Estimated
    hbm_bandwidth_tb=9.0,      # ~9.0 TB/s HBM3e — [B300-ANN] (estimated)
    hbm_capacity_gb=288.0,     # 288 GB — [B300-ANN]
    nvlink_bw_gb=900.0,        # NVLink 5, same topology as B200 — [B300-ANN]
    nvlink_domain=8,
    tdp_w=1_200.0,             # Estimated from power density scaling
    gpu_cost_usd=90_000.0,     # Highly estimated — no public list price
    server_cost_usd=220_000.0,
    depreciation_years=2.0,    # Top price point → fastest refresh cycle
    typical_utilization=0.30,  # Pre-launch; low initial deployment
    opex_overhead_fraction=0.11,
    specs_confirmed=False,     # Pre-datasheet estimate
)

# --- NVIDIA Blackwell NVL72 Rack-Scale Systems ---
# These represent per-GPU specs within the NVL72 rack.
# Key difference vs. SXM: nvlink_domain=72 (full rack NVLink fabric).
# Modeling note: treat an NVL72 rack as a single "node" of 72 GPUs.
# No IB required for intra-rack AllReduce — all traffic stays on NVLink.

NVIDIA_GB200_NVL72_PER_GPU = GPU(
    name="NVIDIA GB200 NVL72 (per-GPU)",
    architecture="Blackwell",
    flops_bf16=2_250.0,        # Same B200 die — [NVL72-WP]
    flops_fp32=150.0,
    hbm_bandwidth_tb=8.0,      # 8.0 TB/s HBM3e per GPU — [NVL72-WP]
    hbm_capacity_gb=192.0,     # 192 GB per GPU — [NVL72-WP]
    nvlink_bw_gb=900.0,        # 1,800 GB/s bidir per GPU to NVLink switch — [NVL72-WP]
    nvlink_domain=72,          # Full rack = one NVLink domain — [NVL72-WP]
    tdp_w=1_000.0,             # ~600 kW rack / 72 GPUs ≈ ~1,000 W/GPU imputed — [NVL72-WP]
    gpu_cost_usd=41_700.0,     # ~$3M rack / 72 GPUs ≈ $41,700/GPU estimated
    server_cost_usd=0.0,       # Chassis cost already amortized into GPU estimate for rack systems
    depreciation_years=2.5,    # Rack-scale system; higher capex → faster refresh
    typical_utilization=0.35,  # Rack-scale ops complexity reduces steady-state utilization
    opex_overhead_fraction=0.13,  # Rack-scale systems have higher integration/facility overhead
    specs_confirmed=True,
)

NVIDIA_GB300_NVL72_PER_GPU = GPU(
    name="NVIDIA GB300 NVL72 (per-GPU)",
    architecture="Blackwell Ultra",
    flops_bf16=2_500.0,        # Same B300 die — [GB300-ANN] (estimated)
    flops_fp32=170.0,
    hbm_bandwidth_tb=9.0,      # ~9.0 TB/s — [GB300-ANN] (estimated)
    hbm_capacity_gb=288.0,     # 288 GB per GPU — [GB300-ANN]
    nvlink_bw_gb=900.0,        # NVLink 5 — [GB300-ANN]
    nvlink_domain=72,          # Same rack-scale NVLink domain
    tdp_w=1_200.0,             # Estimated
    gpu_cost_usd=55_600.0,     # ~$4M rack / 72 GPUs ≈ $55,600/GPU estimated
    server_cost_usd=0.0,
    depreciation_years=2.0,    # Pre-launch; fastest expected refresh
    typical_utilization=0.30,  # Very early deployment
    opex_overhead_fraction=0.13,
    specs_confirmed=False,     # Pre-datasheet estimate
)

# --- AMD CDNA3 Generation ---

AMD_MI300X = GPU(
    name="AMD MI300X",
    architecture="CDNA3",
    flops_bf16=1_307.0,        # 1,307 TFLOP/s BF16 — [MI300X-PB]
    flops_fp32=163.4,          # 163.4 TFLOP/s FP32 — [MI300X-PB]
    hbm_bandwidth_tb=5.3,      # 5.3 TB/s HBM3 — [MI300X-PB]
    hbm_capacity_gb=192.0,     # 192 GB HBM3 — [MI300X-PB]
    nvlink_bw_gb=448.0,        # 896 GB/s bidirectional → 448 GB/s unidirectional — [MI300X-PB]
    nvlink_domain=8,
    tdp_w=750.0,               # [MI300X-PB]
    gpu_cost_usd=15_000.0,     # ~$15k OEM pricing (2024)
    server_cost_usd=100_000.0,
    depreciation_years=3.0,    # Mature CDNA3 gen; standard 3-yr cycle
    typical_utilization=0.40,  # ROCm ecosystem gaps reduce achievable utilization vs CUDA
    opex_overhead_fraction=0.13,  # Higher staff support due to ROCm immaturity vs CUDA
    specs_confirmed=True,
)

AMD_MI325X = GPU(
    name="AMD MI325X",
    architecture="CDNA3",
    flops_bf16=1_307.0,        # Same CDNA3 compute die as MI300X — [MI325X-PB]
    flops_fp32=163.4,
    hbm_bandwidth_tb=6.0,      # 6.0 TB/s HBM3e (vs 5.3 TB/s on MI300X) — [MI325X-PB]
    hbm_capacity_gb=256.0,     # 256 GB HBM3e (vs 192 GB on MI300X) — [MI325X-PB]
    nvlink_bw_gb=448.0,        # Same Infinity Fabric — [MI325X-PB]
    nvlink_domain=8,
    tdp_w=750.0,               # [MI325X-PB]
    gpu_cost_usd=20_000.0,     # ~$20k estimated (premium over MI300X for HBM3e)
    server_cost_usd=105_000.0,
    depreciation_years=3.0,
    typical_utilization=0.38,  # Slightly lower adoption than MI300X
    opex_overhead_fraction=0.14,
    specs_confirmed=True,
)

# --- AMD CDNA4 Generation ---

AMD_MI355X = GPU(
    name="AMD MI355X",
    architecture="CDNA4",
    flops_bf16=2_560.0,        # 2,560 TFLOP/s BF16 dense — [MI355X-PB]
    flops_fp32=320.0,          # Estimated from CDNA4 scaling vs CDNA3
    hbm_bandwidth_tb=8.0,      # 8.0 TB/s HBM3e — [MI355X-PB]
    hbm_capacity_gb=288.0,     # 288 GB HBM3e — [MI355X-PB]
    nvlink_bw_gb=896.0,        # 1,792 GB/s bidir → 896 GB/s unidir (CDNA4 IF) — [MI355X-PB]
    nvlink_domain=8,
    tdp_w=1_000.0,             # Estimated
    gpu_cost_usd=35_000.0,     # Highly estimated — no public list price
    server_cost_usd=130_000.0,
    depreciation_years=2.5,    # CDNA4 is new; faster expected refresh
    typical_utilization=0.35,  # Early deployment; CDNA4 ROCm coverage still maturing
    opex_overhead_fraction=0.14,
    specs_confirmed=True,
)

# --- Legacy / PCIe Variants ---

NVIDIA_H100_PCIE = GPU(
    name="NVIDIA H100 PCIe",
    architecture="Hopper",
    flops_bf16=756.0,          # [H100-WP]
    flops_fp32=51.0,           # [H100-WP]
    hbm_bandwidth_tb=2.0,      # 2.0 TB/s HBM2e — [H100-WP]
    hbm_capacity_gb=80.0,
    nvlink_bw_gb=0.0,          # No NVLink on PCIe variant — [H100-WP]
    nvlink_domain=1,
    tdp_w=350.0,               # [H100-WP]
    gpu_cost_usd=25_000.0,
    server_cost_usd=40_000.0,
    specs_confirmed=True,
)


# =============================================================================
# Network Fabrics
# =============================================================================

IB_HDR = NetworkFabric(
    name="InfiniBand HDR (200 Gb/s)",
    intra_node_bw_gb=450.0,    # NVLink 4 intra-node (unidirectional)
    inter_node_bw_gb=25.0,     # 200 Gb/s HDR → 25 GB/s per port — [IB-HDR]
    latency_us=1.5,            # Typical MPI small-message latency for HDR — [IB-HDR]
    switch_cost_usd=1_500.0,   # Enterprise list price per port estimate (2024)
    ports_per_switch=40,
)

IB_NDR = NetworkFabric(
    name="InfiniBand NDR (400 Gb/s)",
    intra_node_bw_gb=450.0,
    inter_node_bw_gb=50.0,     # 400 Gb/s NDR → 50 GB/s per port — [IB-NDR]
    latency_us=0.8,            # Estimated improvement over HDR
    switch_cost_usd=3_000.0,   # Enterprise list price per port estimate (2024)
    ports_per_switch=64,
)

NVLINK_SWITCH_NVL72 = NetworkFabric(
    name="NVLink 5 Switch (NVL72)",
    intra_node_bw_gb=900.0,    # 1,800 GB/s bidir → 900 GB/s unidirectional per GPU — [NVL72-WP]
    inter_node_bw_gb=50.0,     # IB NDR assumed for inter-rack — [IB-NDR]
    latency_us=0.5,            # NVLink switch latency < IB; estimated
    switch_cost_usd=0.0,       # Bundled into rack system price
    ports_per_switch=72,
)


# =============================================================================
# Pre-built cluster configs for common deployment scenarios
# =============================================================================

CLUSTER_8xH100     = ClusterConfig(gpu=NVIDIA_H100_SXM,          fabric=IB_HDR,  num_nodes=1)
CLUSTER_64xH100    = ClusterConfig(gpu=NVIDIA_H100_SXM,          fabric=IB_NDR,  num_nodes=8)
CLUSTER_8xH200     = ClusterConfig(gpu=NVIDIA_H200_SXM,          fabric=IB_NDR,  num_nodes=1)
CLUSTER_8xB200     = ClusterConfig(gpu=NVIDIA_B200_SXM,          fabric=IB_NDR,  num_nodes=1)
CLUSTER_8xMI300X   = ClusterConfig(gpu=AMD_MI300X,               fabric=IB_HDR,  num_nodes=1)
CLUSTER_64xMI300X  = ClusterConfig(gpu=AMD_MI300X,               fabric=IB_NDR,  num_nodes=8)
CLUSTER_8xMI325X   = ClusterConfig(gpu=AMD_MI325X,               fabric=IB_NDR,  num_nodes=1)
CLUSTER_8xMI355X   = ClusterConfig(gpu=AMD_MI355X,               fabric=IB_NDR,  num_nodes=1)

# NVL72 clusters: one "node" = one full rack of 72 GPUs on NVLink fabric
CLUSTER_NVL72_GB200 = ClusterConfig(
    gpu=NVIDIA_GB200_NVL72_PER_GPU,
    fabric=NVLINK_SWITCH_NVL72,
    num_nodes=1,
    gpus_per_node=72,
)
CLUSTER_NVL72_GB300 = ClusterConfig(
    gpu=NVIDIA_GB300_NVL72_PER_GPU,
    fabric=NVLINK_SWITCH_NVL72,
    num_nodes=1,
    gpus_per_node=72,
)


# All GPU entries for iteration / comparison plots
ALL_GPUS = [
    NVIDIA_H100_SXM,
    NVIDIA_H200_SXM,
    NVIDIA_B200_SXM,
    NVIDIA_B300_SXM,
    NVIDIA_GB200_NVL72_PER_GPU,
    NVIDIA_GB300_NVL72_PER_GPU,
    AMD_MI300X,
    AMD_MI325X,
    AMD_MI355X,
]

# Confirmed (non-estimated) GPUs only
CONFIRMED_GPUS = [g for g in ALL_GPUS if g.specs_confirmed]
