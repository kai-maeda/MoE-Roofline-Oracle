"""
Cost modeling: CapEx, TCO, and cost-per-token for MoE inference clusters.

This is the InferenceX-style layer — translating performance metrics into
economic outcomes. The goal is to answer: "What does it cost to serve
X tokens/day at Y latency SLA, on H100 vs MI300X?"

Methodology:
  - CapEx: GPU + server chassis + networking switches
  - OpEx: power (at datacenter PUE-adjusted rate) + non-power overhead
  - TCO: amortized CapEx over hardware lifetime + total OpEx
  - Cost/token: annual TCO / (utilization × throughput × seconds_per_year)

=============================================================================
ECONOMIC ASSUMPTION CITATIONS
=============================================================================

Hardware depreciation period (3 years)
  [Uptime23]   Uptime Institute. "Global Data Center Survey 2023."
               Typical hyperscaler GPU depreciation schedule: 3-4 years.
               We use 3 years (conservative / standard for fast-moving hardware).

Power cost ($0.07/kWh)
  [EIA24]      U.S. Energy Information Administration. "Electric Power Monthly,
               Table 5.6: Average Retail Price of Electricity, Industrial."
               (2024). Industrial/commercial datacenter power rates: $0.05-0.10/kWh.
               We use $0.07/kWh as a midpoint for large datacenter contracts.
               Note: colocation and hyperscaler PPA rates may be lower ($0.04-0.06).

PUE = 1.3 (Power Usage Effectiveness)
  [GreenGrid]  The Green Grid. "PUE: A Comprehensive Examination of the Metric."
               White Paper #49 (2012). Industry average PUE for large datacenters
               has trended toward 1.2-1.4. We use 1.3 as a conservative midpoint.
               Source also: Uptime Institute Global Survey 2023 (avg PUE 1.58
               globally; hyperscaler campuses achieve 1.1-1.2).

Cluster utilization = 45%
  [SemiAnalysis24] SemiAnalysis. "AI Datacenter Economics." (2024).
                   Inference clusters at scale typically operate at 40-60% GPU
                   utilization when averaged over 24h (demand peaks, cold starts,
                   model switching). We use 45% as a conservative baseline.
                   Higher utilization assumptions would lower cost/token estimates.

Non-power OpEx = 10% of CapEx/year
  [Uptime23]   Ibid. Operational overhead including maintenance contracts,
               network transit, staff, and facility fees typically adds
               8-15% of CapEx per year on top of power costs.

Hardware availability = 97%
  [AWS23]      Amazon Web Services. "Amazon EC2 SLA." (2023). Large GPU
               clusters experience ~1-3% downtime annually from hardware
               failures, driver issues, and planned maintenance.

Installation cost = 5% of hardware CapEx
  [Gartner22]  Gartner. "IT Infrastructure Cost Benchmarking." (2022).
               Rack integration, cabling, burn-in testing, and shipping
               typically adds 3-7% to hardware list price. We use 5%.
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from oracle.hardware import GPU, ClusterConfig, NetworkFabric
from oracle.roofline import InferenceProfile, RooflineResult
from oracle.parallelism import ParallelismConfig, ParallelismOverhead


# ---------------------------------------------------------------------------
# Economic assumptions (adjustable)
# ---------------------------------------------------------------------------

@dataclass
class EconomicAssumptions:
    """
    Operator-level economic assumptions: things that depend on WHERE you build,
    not WHAT hardware you buy. Hardware-specific defaults (depreciation, utilization,
    OpEx overhead) live on the GPU dataclass in hardware.py.
    """
    power_cost_per_kwh: float = 0.07  # $0.07/kWh datacenter rate — [EIA24]
    pue: float = 1.3                  # Power Usage Effectiveness — [GreenGrid]
    discount_rate: float = 0.08       # 8% capital discount rate (standard WACC estimate)

    @property
    def seconds_per_year(self) -> float:
        return 365.25 * 24 * 3600


DEFAULT_ASSUMPTIONS = EconomicAssumptions()


# ---------------------------------------------------------------------------
# CapEx breakdown
# ---------------------------------------------------------------------------

@dataclass
class CapExBreakdown:
    gpu_cost_usd: float
    server_cost_usd: float
    network_cost_usd: float
    installation_cost_usd: float   # ~5% of hardware

    @property
    def total_usd(self) -> float:
        return (
            self.gpu_cost_usd +
            self.server_cost_usd +
            self.network_cost_usd +
            self.installation_cost_usd
        )

    @property
    def gpu_fraction(self) -> float:
        return self.gpu_cost_usd / self.total_usd

    def summary(self) -> dict:
        return {
            "GPU hardware": f"${self.gpu_cost_usd:,.0f}",
            "Server chassis": f"${self.server_cost_usd:,.0f}",
            "Networking": f"${self.network_cost_usd:,.0f}",
            "Installation (5%)": f"${self.installation_cost_usd:,.0f}",
            "Total CapEx": f"${self.total_usd:,.0f}",
            "GPU fraction": f"{self.gpu_fraction:.1%}",
        }


def compute_capex(cluster: ClusterConfig) -> CapExBreakdown:
    gpu_cost = cluster.capex_gpu_usd
    server_cost = cluster.capex_server_usd
    network_cost = cluster.capex_network_usd
    install_cost = 0.05 * (gpu_cost + server_cost + network_cost)  # 5% install — [Gartner22]
    return CapExBreakdown(
        gpu_cost_usd=gpu_cost,
        server_cost_usd=server_cost,
        network_cost_usd=network_cost,
        installation_cost_usd=install_cost,
    )


# ---------------------------------------------------------------------------
# TCO and cost-per-token
# ---------------------------------------------------------------------------

@dataclass
class TCOResult:
    cluster: ClusterConfig
    capex: CapExBreakdown
    assumptions: EconomicAssumptions

    # Annual costs
    annual_power_cost_usd: float = 0.0
    annual_opex_usd: float = 0.0

    def __post_init__(self):
        self._compute()

    def _compute(self):
        gpu = self.cluster.gpu
        # Power: GPU TDP × PUE × hours/year × $/kWh
        total_kw = self.cluster.total_tdp_kw
        self.annual_power_cost_usd = (
            total_kw *
            self.assumptions.pue *
            8760 *  # hours/year
            self.assumptions.power_cost_per_kwh
        )
        # Non-power OpEx: hardware-specific overhead fraction (varies by ecosystem maturity)
        self.annual_opex_usd = (
            self.capex.total_usd *
            gpu.opex_overhead_fraction +
            self.annual_power_cost_usd
        )

    @property
    def total_tco_usd(self) -> float:
        """Total cost of ownership over hardware lifetime."""
        capex = self.capex.total_usd
        opex = self.annual_opex_usd * self.cluster.gpu.depreciation_years
        return capex + opex

    @property
    def annual_tco_usd(self) -> float:
        capex_annual = self.capex.total_usd / self.cluster.gpu.depreciation_years
        return capex_annual + self.annual_opex_usd

    @property
    def cost_per_gpu_hour_usd(self) -> float:
        total_gpu_hours = (
            self.cluster.total_gpus *
            8760 *
            self.cluster.gpu.depreciation_years
        )
        return self.total_tco_usd / total_gpu_hours if total_gpu_hours > 0 else 0.0

    def cost_per_million_tokens(
        self,
        tokens_per_second_per_gpu: float,
    ) -> float:
        """
        Cost per million output tokens, accounting for utilization.

        Args:
            tokens_per_second_per_gpu: Achieved decode throughput per GPU
        """
        # hardware_availability = 0.97 (97%) — [AWS23]
        eff = 0.97 * self.cluster.gpu.typical_utilization
        tokens_per_year = (
            tokens_per_second_per_gpu *
            self.assumptions.seconds_per_year *
            eff *
            self.cluster.total_gpus
        )
        if tokens_per_year == 0:
            return float('inf')
        return (self.annual_tco_usd / tokens_per_year) * 1e6

    def summary(self) -> dict:
        gpu = self.cluster.gpu
        return {
            "Cluster": f"{self.cluster.total_gpus}x {gpu.name}",
            "CapEx": f"${self.capex.total_usd:,.0f}",
            "Annual power": f"${self.annual_power_cost_usd:,.0f}",
            "Annual TCO": f"${self.annual_tco_usd:,.0f}",
            f"Total TCO ({gpu.depreciation_years:.0f}yr)":
                f"${self.total_tco_usd:,.0f}",
            "Cost/GPU-hour": f"${self.cost_per_gpu_hour_usd:.2f}",
        }


# ---------------------------------------------------------------------------
# Pareto frontier: cost vs latency trade-off
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """One point on the cost-latency Pareto frontier."""
    parallelism_config: ParallelismConfig
    num_gpus: int
    cost_per_million_tokens_usd: float
    inter_token_latency_ms: float
    time_to_first_token_ms: float
    tokens_per_second: float
    mfu: float
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = str(self.parallelism_config)


def build_pareto_frontier(
    gpu: GPU,
    fabric: NetworkFabric,
    inference_profiles: List[Tuple[ParallelismConfig, InferenceProfile, ParallelismOverhead]],
    assumptions: EconomicAssumptions = None,
) -> List[ParetoPoint]:
    """
    Build Pareto frontier of cost vs latency across parallelism configs.

    Filters dominated points (higher cost AND higher latency than another point).
    """
    if assumptions is None:
        assumptions = DEFAULT_ASSUMPTIONS

    points = []
    for cfg, profile, overhead in inference_profiles:
        num_gpus = cfg.total_gpus
        cluster = ClusterConfig(gpu=gpu, fabric=fabric, num_nodes=max(1, num_gpus // 8))
        capex = compute_capex(cluster)
        tco = TCOResult(cluster=cluster, capex=capex, assumptions=assumptions)

        # Apply parallelism efficiency to throughput
        eff = overhead.effective_efficiency(profile.decode_result.bottleneck_time_s)
        tok_s = profile.tokens_per_second_decode * eff

        cost_per_m = tco.cost_per_million_tokens(tok_s / num_gpus)
        itl_ms = profile.inter_token_latency_ms / eff if eff > 0 else float('inf')
        ttft_ms = profile.time_to_first_token_ms / eff if eff > 0 else float('inf')

        points.append(ParetoPoint(
            parallelism_config=cfg,
            num_gpus=num_gpus,
            cost_per_million_tokens_usd=cost_per_m,
            inter_token_latency_ms=itl_ms,
            time_to_first_token_ms=ttft_ms,
            tokens_per_second=tok_s,
            mfu=profile.decode_result.mfu_theoretical * eff,
        ))

    # Filter dominated points
    pareto = []
    for p in points:
        dominated = any(
            (q.cost_per_million_tokens_usd <= p.cost_per_million_tokens_usd and
             q.inter_token_latency_ms <= p.inter_token_latency_ms and
             q is not p)
            for q in points
        )
        if not dominated:
            pareto.append(p)

    pareto.sort(key=lambda x: x.inter_token_latency_ms)
    return pareto


# ---------------------------------------------------------------------------
# Hardware comparison utility
# ---------------------------------------------------------------------------

@dataclass
class HardwareComparison:
    """Side-by-side cost comparison between two GPU options."""
    gpu_a: GPU
    gpu_b: GPU
    tco_a: TCOResult
    tco_b: TCOResult
    tok_s_a: float
    tok_s_b: float
    assumptions: EconomicAssumptions

    @property
    def cost_per_m_a(self) -> float:
        return self.tco_a.cost_per_million_tokens(self.tok_s_a / self.tco_a.cluster.total_gpus)

    @property
    def cost_per_m_b(self) -> float:
        return self.tco_b.cost_per_million_tokens(self.tok_s_b / self.tco_b.cluster.total_gpus)

    @property
    def cost_winner(self) -> GPU:
        return self.gpu_a if self.cost_per_m_a < self.cost_per_m_b else self.gpu_b

    @property
    def cost_advantage_pct(self) -> float:
        """% cheaper for the winning GPU."""
        if self.cost_per_m_a < self.cost_per_m_b:
            return (self.cost_per_m_b - self.cost_per_m_a) / self.cost_per_m_b * 100
        return (self.cost_per_m_a - self.cost_per_m_b) / self.cost_per_m_a * 100

    def summary(self) -> dict:
        return {
            self.gpu_a.name: {
                "CapEx": f"${self.tco_a.capex.total_usd:,.0f}",
                "Annual TCO": f"${self.tco_a.annual_tco_usd:,.0f}",
                "Cost/GPU-hr": f"${self.tco_a.cost_per_gpu_hour_usd:.2f}",
                "Tok/s/GPU": f"{self.tok_s_a / self.tco_a.cluster.total_gpus:.0f}",
                "$/M tokens": f"${self.cost_per_m_a:.2f}",
            },
            self.gpu_b.name: {
                "CapEx": f"${self.tco_b.capex.total_usd:,.0f}",
                "Annual TCO": f"${self.tco_b.annual_tco_usd:,.0f}",
                "Cost/GPU-hr": f"${self.tco_b.cost_per_gpu_hour_usd:.2f}",
                "Tok/s/GPU": f"{self.tok_s_b / self.tco_b.cluster.total_gpus:.0f}",
                "$/M tokens": f"${self.cost_per_m_b:.2f}",
            },
            "Winner": self.cost_winner.name,
            "Cost advantage": f"{self.cost_advantage_pct:.1f}% cheaper",
        }
