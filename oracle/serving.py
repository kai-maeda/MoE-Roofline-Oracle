"""
Continuous-batching throughput-latency model using M/D/1 queueing theory.

In production LLM serving with continuous batching, requests arrive according
to a Poisson process. The GPU decodes a batch of B concurrent requests per
step, each step taking decode_step_s seconds. A new request queues until the
system has capacity to prefill it.

Service rate (max sustainable throughput):
    μ = batch_size / (osl × decode_step_s)            [req/s]

Utilization:
    ρ = λ / μ                                          [dimensionless, 0–1]

M/D/1 mean queue wait (Pollaczek-Khinchine formula, deterministic service):
    W_q = ρ / (2μ(1 − ρ))                             [seconds]

Mean TTFT under load:
    E[TTFT] = W_q + ttft_base

P95 TTFT (exponential tail approximation for M/D/1 waiting time):
    P(W_q > t) ≈ ρ · exp(−2(1−ρ)μt)
    → TTFT_P95 ≈ ttft_base + ln(20ρ) / (2(1−ρ)μ)

Max QPS at TTFT SLA T:
    Solve W_q = T − ttft_base for λ:
    ρ* = 2μ(T−ttft_base) / (1 + 2μ(T−ttft_base))
    λ* = ρ* · μ

=============================================================================
CITATIONS
=============================================================================

  [Kleinrock75]  Kleinrock, L. "Queueing Systems, Vol. 1: Theory." Wiley,
                 1975. Chapter 4 — M/D/1 mean waiting time via
                 Pollaczek-Khinchine mean value formula.

  [Gross98]      Gross, D., Harris, C. "Fundamentals of Queueing Theory."
                 3rd ed. Wiley, 1998. M/D/1 waiting time distribution and
                 exponential tail approximation.

  [Agrawal24]    Agrawal et al. "SARATHI-SERVE: Efficient LLM Inference by
                 Piggybacking Decodes with Chunked Prefills." OSDI 2024.
                 Continuous batching throughput model; prefill-decode
                 interleaving in production serving systems.
=============================================================================
"""

from dataclasses import dataclass
from typing import List
import math
import numpy as np


@dataclass
class ServingPoint:
    """
    TTFT and throughput at a single arrival rate operating point.
    All TTFT values include the base prefill latency.
    """
    arrival_rate_rps: float    # λ — request arrival rate (req/s)
    utilization: float         # ρ = λ/μ
    mean_ttft_ms: float        # E[TTFT] including queue wait
    p95_ttft_ms: float         # P95 TTFT (exponential tail approx)
    max_throughput_rps: float  # μ = batch_size / (osl × decode_step_s)


def _mu(batch_size: int, osl: int, decode_step_s: float) -> float:
    """Maximum sustainable throughput in req/s."""
    denom = osl * decode_step_s
    return batch_size / denom if denom > 0 else 0.0


def compute_serving_point(
    arrival_rate_rps: float,
    ttft_base_ms: float,
    decode_step_ms: float,
    osl: int,
    batch_size: int,
) -> ServingPoint:
    """TTFT statistics at a given arrival rate."""
    ttft_base_s = ttft_base_ms / 1e3
    decode_step_s = decode_step_ms / 1e3
    mu = _mu(batch_size, osl, decode_step_s)

    if mu == 0:
        return ServingPoint(arrival_rate_rps, 0.0, ttft_base_ms, ttft_base_ms, 0.0)

    rho = min(arrival_rate_rps / mu, 0.9999)

    # M/D/1 Pollaczek-Khinchine mean waiting time [Kleinrock75]
    W_q_s = rho / (2 * mu * (1 - rho))
    mean_ttft_ms = (W_q_s + ttft_base_s) * 1e3

    # P95 via exponential tail: P(W_q > t) ≈ ρ·exp(−2(1−ρ)μt) [Gross98]
    if rho > 0.05:
        t95_s = math.log(20 * rho) / (2 * (1 - rho) * mu)
    else:
        t95_s = 0.0
    p95_ttft_ms = (t95_s + ttft_base_s) * 1e3

    return ServingPoint(
        arrival_rate_rps=arrival_rate_rps,
        utilization=rho,
        mean_ttft_ms=mean_ttft_ms,
        p95_ttft_ms=p95_ttft_ms,
        max_throughput_rps=mu,
    )


def max_qps_at_sla(
    ttft_sla_ms: float,
    ttft_base_ms: float,
    decode_step_ms: float,
    osl: int,
    batch_size: int,
) -> float:
    """
    Maximum arrival rate (req/s) such that mean TTFT ≤ ttft_sla_ms.

    Derived by solving W_q = slack for ρ then λ = ρ·μ:
        ρ* = 2μ·slack / (1 + 2μ·slack)
        λ* = ρ* · μ

    Returns 0 if the SLA is tighter than the zero-load TTFT (ttft_base).
    """
    slack_s = (ttft_sla_ms - ttft_base_ms) / 1e3
    if slack_s <= 0:
        return 0.0
    decode_step_s = decode_step_ms / 1e3
    mu = _mu(batch_size, osl, decode_step_s)
    if mu == 0:
        return 0.0
    a = 2 * mu * slack_s
    rho_star = a / (1 + a)
    return rho_star * mu


def sweep_arrival_rates(
    ttft_base_ms: float,
    decode_step_ms: float,
    osl: int,
    batch_size: int,
    n_points: int = 80,
) -> List[ServingPoint]:
    """Sweep λ from near-zero to 99% utilization."""
    decode_step_s = decode_step_ms / 1e3
    mu = _mu(batch_size, osl, decode_step_s)
    if mu == 0:
        return []
    rates = np.linspace(0.01 * mu, 0.99 * mu, n_points)
    return [
        compute_serving_point(r, ttft_base_ms, decode_step_ms, osl, batch_size)
        for r in rates
    ]
