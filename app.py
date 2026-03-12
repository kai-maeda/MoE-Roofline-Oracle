"""
MoE Roofline Oracle — Interactive Dashboard
InferenceX-style interface for filtering performance and cost results
across hardware, models, parallelism configurations, and serving regimes.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from oracle.hardware import (
    ALL_GPUS, IB_NDR, IB_HDR, NVLINK_SWITCH_NVL72,
    ClusterConfig, NetworkFabric,
)
from oracle.workload import ALL_MODELS, ModelStats, moe_load_imbalance_factor, gpu_memory_required_bytes
from oracle.roofline import roofline, InferenceProfile
from oracle.parallelism import (
    ParallelismConfig, compute_parallelism_overhead, sweep_parallelism_configs
)
from oracle.cost import (
    EconomicAssumptions, DEFAULT_ASSUMPTIONS,
    compute_capex, TCOResult, build_pareto_frontier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MoE Roofline Oracle",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ MoE Roofline Oracle")
st.caption(
    "Analytical performance & cost simulator for MoE LLM inference. "
    "Inspired by SemiAnalysis InferenceX methodology."
)

# ─────────────────────────────────────────────────────────────────────────────
# Top controls
# ─────────────────────────────────────────────────────────────────────────────

gpu_name_map  = {g.name: g for g in ALL_GPUS}
selected_gpus = ALL_GPUS
model_name_map = {m.name: m for m in ALL_MODELS}

ISL_OSL_PRESETS = {
    "8k / 1k  — RAG / summarization":  (8192, 1024),
    "1k / 1k  — Chatbot":              (1024, 1024),
    "1k / 8k  — Reasoning / CoT":      (1024, 8192),
}

DTYPE_OPTIONS = {
    "BF16": {"dtype_bytes": 2.0, "compute_scale": 1.0},
    "FP8":  {"dtype_bytes": 1.0, "compute_scale": 2.0},
    "FP4":  {"dtype_bytes": 0.5, "compute_scale": 4.0},
}

PRECISION_DASH = {"BF16": "solid", "FP8": "dash", "FP4": "dot"}

COST_PROVIDERS = {
    "Hyperscaler":      {"power_cost": 0.04, "pue": 1.12, "desc": "Long-term PPA, campus-grade cooling (Google/MS/AWS)"},
    "Neocloud":         {"power_cost": 0.07, "pue": 1.25, "desc": "CoreWeave / Lambda / Crusoe tier"},
    "3yr Colo Rental":  {"power_cost": 0.10, "pue": 1.40, "desc": "Shared colocation facility"},
}

col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
with col1:
    selected_model_name = st.selectbox("Model", options=list(model_name_map.keys()), index=1)
with col2:
    preset = st.selectbox("Workload (ISL / OSL)", list(ISL_OSL_PRESETS.keys()), index=1)
with col3:
    provider = st.selectbox("Cost provider", list(COST_PROVIDERS.keys()), index=1)
with col4:
    selected_precisions = st.multiselect(
        "Precision", options=list(DTYPE_OPTIONS.keys()), default=["BF16", "FP8"],
    )
with col5:
    target_utilization = st.slider(
        "Cluster utilization",
        min_value=5, max_value=100, value=45, step=5,
        help=(
            "Expected fraction of time the cluster is actively serving requests. "
            "Overrides per-GPU defaults. "
            "Low-traffic startups: 5–20%. At-scale deployments: 40–60%. "
            "Higher utilization → lower $/M tokens."
        ),
    ) / 100.0

model = model_name_map[selected_model_name]
prefill_seq_len, decode_seq_len = ISL_OSL_PRESETS[preset]
prov       = COST_PROVIDERS[provider]
power_cost = prov["power_cost"]
pue        = prov["pue"]
st.caption(prov["desc"] + f"  ·  ${power_cost}/kWh  ·  PUE {pue}")

econ = EconomicAssumptions(
    power_cost_per_kwh=power_cost,
    pue=pue,
)

# ─────────────────────────────────────────────────────────────────────────────
# Guard: need at least one GPU selected
# ─────────────────────────────────────────────────────────────────────────────

fabric = IB_NDR  # default fabric for cost calculations

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run roofline + cost for a GPU
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_analysis(
    gpu_name, model_name, prefill_seq_len, decode_seq_len,
    batch_size, tp, pp, ep, power_cost, pue,
    dtype_bytes=2.0, dtype_compute_scale=1.0,
):
    from dataclasses import replace as dc_replace
    gpu   = gpu_name_map[gpu_name]
    model = dc_replace(model_name_map[model_name],
                       weight_dtype_bytes=dtype_bytes,
                       kv_cache_dtype_bytes=dtype_bytes)
    par   = ParallelismConfig(tp=tp, pp=pp, ep=ep)
    fab   = IB_NDR
    eco   = EconomicAssumptions(
        power_cost_per_kwh=power_cost,
        pue=pue,
    )

    # Roofline
    profile = InferenceProfile(
        model=model,
        gpu=gpu,
        prefill_seq_len=prefill_seq_len,
        prefill_batch_size=batch_size,
        decode_seq_len=decode_seq_len,
        decode_batch_size=batch_size,
        dtype_compute_scale=dtype_compute_scale,
    )

    # Parallelism overhead
    prefill_stats = ModelStats(model=model, seq_len=prefill_seq_len,
                               batch_size=batch_size, phase="prefill")
    decode_stats  = ModelStats(model=model, seq_len=decode_seq_len,
                               batch_size=1, phase="decode")

    par_overhead_decode = compute_parallelism_overhead(
        config=par,
        model_d=model.d_model,
        n_layers=model.n_layers,
        n_moe_layers=model.n_moe_layers,
        n_experts_active=model.n_experts_active,
        batch_size=1,
        seq_len=decode_seq_len,
        fabric=fab,
        phase="decode",
    )
    decode_eff = par_overhead_decode.effective_efficiency(
        profile.decode_result.bottleneck_time_s
    )

    # Cost
    gpus_per_node = 72 if "NVL72" in gpu.name else 8
    num_nodes     = max(1, par.total_gpus // gpus_per_node)
    cluster = ClusterConfig(gpu=gpu, fabric=fab,
                            num_nodes=num_nodes, gpus_per_node=gpus_per_node)
    capex = compute_capex(cluster)
    tco   = TCOResult(cluster=cluster, capex=capex, assumptions=eco)

    tok_s_decode = profile.tokens_per_second_decode * decode_eff * gpu.spec_decode_speedup
    itl_ms_eff   = profile.inter_token_latency_ms / gpu.spec_decode_speedup
    cost_per_m   = tco.cost_per_million_tokens(
        tok_s_decode / max(1, cluster.total_gpus)
    )

    return {
        "gpu":              gpu.name,
        "architecture":     gpu.architecture,
        "confirmed":        gpu.specs_confirmed,
        # Prefill
        "prefill_bottleneck":  profile.prefill_result.bottleneck,
        "prefill_mfu":         profile.prefill_result.mfu_theoretical,
        "prefill_intensity":   profile.prefill_result.arithmetic_intensity,
        "ttft_ms":             profile.time_to_first_token_ms,
        # Decode (spec_decode_speedup applied to itl_ms and tok_s)
        "decode_bottleneck":   profile.decode_result.bottleneck,
        "decode_mfu":          profile.decode_result.mfu_theoretical,
        "decode_intensity":    profile.decode_result.arithmetic_intensity,
        "itl_ms":              itl_ms_eff,
        "tok_s_decode":        tok_s_decode,
        "tok_s_per_gpu":       tok_s_decode / max(1, cluster.total_gpus),
        "decode_efficiency":   decode_eff,
        # Cost
        "capex_m":             capex.total_usd / 1e6,
        "tco_annual_m":        tco.annual_tco_usd / 1e6,
        "cost_per_m_tokens":   cost_per_m,
        "cost_per_gpu_hr":     tco.cost_per_gpu_hour_usd,
        # Raw objects for plots
        "_profile":   profile,
        "_tco":       tco,
        "_gpu_obj":   gpu,
    }

@st.cache_data(show_spinner=False)
def sweep_pareto_configs(gpu_name, model_name, prefill_seq_len, decode_seq_len, power_cost, pue,
                         dtype_bytes=2.0, dtype_compute_scale=1.0, target_utilization=0.45):
    from dataclasses import replace as dc_replace
    gpu   = dc_replace(gpu_name_map[gpu_name], typical_utilization=target_utilization)
    model = dc_replace(model_name_map[model_name],
                       weight_dtype_bytes=dtype_bytes,
                       kv_cache_dtype_bytes=dtype_bytes)
    fab   = IB_NDR
    eco   = EconomicAssumptions(power_cost_per_kwh=power_cost, pue=pue)
    gpus_per_node = 72 if "NVL72" in gpu.name else 8

    configs = [
        ParallelismConfig(tp=tp, pp=pp, ep=ep)
        for tp in [1, 2, 4, 8]
        for ep in [1, 2, 4, 8, 16]
        for pp in [1, 2, 4, 8]
        if ep <= model.n_experts
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # Compute FFN fraction of decode memory once (invariant across batch / cfg).
    # MoE imbalance only affects the expert FFN layers; attention and comms are
    # unaffected by routing.  ffn_fraction scales the penalty correctly.
    _base_decode = ModelStats(model=model, seq_len=decode_seq_len, batch_size=1, phase="decode")
    _ffn_bytes   = sum(l.ffn.bytes_total  for l in _base_decode.layer_stats)
    _tot_bytes   = sum(l.total_bytes      for l in _base_decode.layer_stats)
    ffn_fraction = _ffn_bytes / _tot_bytes if _tot_bytes > 0 else 0.0

    all_points = []
    for cfg in configs:
        num_gpus  = cfg.total_gpus
        num_nodes = max(1, num_gpus // gpus_per_node)
        cluster   = ClusterConfig(gpu=gpu, fabric=fab, num_nodes=num_nodes, gpus_per_node=gpus_per_node)
        capex     = compute_capex(cluster)
        tco       = TCOResult(cluster=cluster, capex=capex, assumptions=eco)

        for batch_size in batch_sizes:
            # Memory feasibility: skip configs that don't fit in HBM
            mem_required = gpu_memory_required_bytes(
                model, cfg.tp, cfg.ep, cfg.pp, batch_size, decode_seq_len,
            )
            if mem_required > gpu.hbm_capacity_gb * 1e9:
                continue

            profile = InferenceProfile(
                model=model, gpu=gpu,
                prefill_seq_len=prefill_seq_len, prefill_batch_size=batch_size,
                decode_seq_len=decode_seq_len,   decode_batch_size=batch_size,
                dtype_compute_scale=dtype_compute_scale,
            )
            overhead = compute_parallelism_overhead(
                config=cfg, model_d=model.d_model,
                n_layers=model.n_layers, n_moe_layers=model.n_moe_layers,
                n_experts_active=model.n_experts_active,
                batch_size=batch_size, seq_len=decode_seq_len,
                fabric=fab, phase="decode",
            )

            decode_speedup  = cfg.tp * cfg.ep
            prefill_speedup = cfg.tp * cfg.ep * cfg.pp

            # Use batch-specific forward-pass time for ITL.
            batch_compute_s   = profile.decode_result.bottleneck_time_s / decode_speedup
            prefill_compute_s = (profile.time_to_first_token_ms  / 1e3) / prefill_speedup

            comm_s = overhead.total_comm_time_s

            # MoE expert load imbalance: straggler expert slows the FFN portion.
            # Only the FFN is expert-routed; attention runs uniformly on all tokens.
            # effective_imbalance = 1 + ffn_fraction × (raw_factor − 1)
            # Source: moe_load_imbalance_factor() in oracle/workload.py.
            # n_ep_groups = actual EP parallelism degree (bins for imbalance).
            # Using total expert count here would treat every expert as its
            # own rank, which is only correct at EP=n_experts.  At EP=1
            # there is no inter-GPU contention so factor must be 1.0.
            n_ep_groups = min(cfg.ep, model.n_experts)
            imbalance = moe_load_imbalance_factor(
                batch_size, n_ep_groups, model.n_experts_active
            )
            eff_imbalance = 1.0 + ffn_fraction * (imbalance - 1.0)
            batch_compute_s_adj = batch_compute_s * eff_imbalance

            batch_step_s = batch_compute_s_adj + comm_s
            itl_ms  = batch_step_s * 1e3
            ttft_ms = (prefill_compute_s + comm_s) * 1e3

            tok_s = batch_size / batch_step_s if batch_step_s > 0 else 0.0

            eff = batch_compute_s_adj / batch_step_s if batch_step_s > 0 else 1.0
            cost_per_m = tco.cost_per_million_tokens(tok_s / max(1, num_gpus))

            all_points.append({
                "tp": cfg.tp, "ep": cfg.ep, "pp": cfg.pp,
                "batch_size": batch_size,
                "num_gpus": num_gpus,
                "cost_per_m": cost_per_m,
                "itl_ms": itl_ms,
                "ttft_ms": ttft_ms,
                "tok_s": tok_s,
                "eff": eff,
            })

    def _pareto(points, lat_key):
        frontier = [
            p for p in points
            if not any(
                q["cost_per_m"] <= p["cost_per_m"] and
                q[lat_key]      <= p[lat_key] and
                q is not p
                for q in points
            )
        ]
        frontier.sort(key=lambda x: x[lat_key])
        return frontier

    return all_points, _pareto(all_points, "itl_ms"), _pareto(all_points, "ttft_ms")


# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────

colors = px.colors.qualitative.Plotly

tab_pareto, tab_roofline = st.tabs([
    "🎯 Pareto Frontier",
    "📈 Roofline Chart",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Roofline Chart
# ─────────────────────────────────────────────────────────────────────────────

with tab_roofline:
    with st.expander("Hardware specs reference"):
        spec_rows = []
        for g in selected_gpus:
            row = {
                "GPU": g.name,
                "Arch": g.architecture,
                "BF16 TFLOPS": f"{g.flops_bf16:.0f}",
                "FP8 TFLOPS": f"{g.flops_bf16 * 2:.0f}",
                "FP4 TFLOPS": f"{g.flops_bf16 * 4:.0f}" if g.supports_fp4 else "—",
                "HBM BW (TB/s)": f"{g.hbm_bandwidth_tb:.1f}",
                "HBM Cap (GB)": f"{g.hbm_capacity_gb:.0f}",
                "Ridge BF16": f"{g.arithmetic_intensity_ridge:.0f}",
                "TDP (W)": f"{g.tdp_w:.0f}",
                "$/TFLOP": f"${g.cost_per_tflop:.1f}",
                "SW Eff Compute": f"{g.sw_efficiency_compute:.0%}",
                "SW Eff BW": f"{g.sw_efficiency_bw:.0%}",
                "Confirmed": "✓" if g.specs_confirmed else "~ est.",
            }
            spec_rows.append(row)
        st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    fig = go.Figure()

    from dataclasses import replace as dc_replace

    # Compute all intensities first so x_range can cover all vertical lines
    _all_intensities = []
    for prec_name in (selected_precisions or ["BF16"]):
        dtype_bytes_p = DTYPE_OPTIONS[prec_name]["dtype_bytes"]
        _m = dc_replace(model, weight_dtype_bytes=dtype_bytes_p, kv_cache_dtype_bytes=dtype_bytes_p)
        pf = ModelStats(model=_m, seq_len=prefill_seq_len, batch_size=8, phase="prefill")
        dc_stats = ModelStats(model=_m, seq_len=decode_seq_len, batch_size=8, phase="decode")
        _all_intensities += [
            dc_stats.attention_intensity(), dc_stats.ffn_intensity(),
            pf.attention_intensity(), pf.ffn_intensity(),
        ]
    max_intensity = max(_all_intensities) if _all_intensities else 1e4
    x_max_log = np.ceil(np.log10(max_intensity * 3))  # extend 3× past rightmost line
    x_range = np.logspace(-1, x_max_log, 1000)

    # y ceiling = highest peak across all selected (gpu × precision) combinations
    max_peak = max(
        (g.flops_bf16 * DTYPE_OPTIONS[p]["compute_scale"] * g.sw_efficiency_compute)
        for g in selected_gpus
        for p in (selected_precisions or ["BF16"])
        if p != "FP4" or g.supports_fp4
    )
    y_min, y_max = 1.0, max_peak * 1.2

    for idx, (gpu, color) in enumerate(zip(selected_gpus, colors)):
        short = gpu.name.replace("NVIDIA ", "").replace("AMD ", "")
        for prec_name in (selected_precisions or ["BF16"]):
            if prec_name == "FP4" and not gpu.supports_fp4:
                continue
            compute_scale = DTYPE_OPTIONS[prec_name]["compute_scale"]
            dash = PRECISION_DASH[prec_name]
            peak_compute = gpu.flops_bf16 * gpu.sw_efficiency_compute * compute_scale
            peak_bw      = gpu.hbm_bandwidth_tb * gpu.sw_efficiency_bw
            y_roof = np.minimum(peak_compute, peak_bw * x_range)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_roof,
                mode="lines",
                name=f"{short} ({prec_name})",
                line=dict(color=color, width=2, dash=dash),
                legendgroup=f"{gpu.name}_{prec_name}",
                showlegend=True,
            ))

    # Intensity lines per selected precision — same op colors, dash encodes precision
    OP_COLORS = {
        "Attn decode":  "#F0E68C",
        "FFN decode":   "#FFA07A",
        "Attn prefill": "#00CED1",
        "FFN prefill":  "#98FB98",
    }
    for prec_name in (selected_precisions or ["BF16"]):
        dtype_bytes_p = DTYPE_OPTIONS[prec_name]["dtype_bytes"]
        prec_dash = PRECISION_DASH[prec_name]
        _m = dc_replace(model, weight_dtype_bytes=dtype_bytes_p, kv_cache_dtype_bytes=dtype_bytes_p)
        pf = ModelStats(model=_m, seq_len=prefill_seq_len, batch_size=8, phase="prefill")
        dc_s = ModelStats(model=_m, seq_len=decode_seq_len, batch_size=8, phase="decode")
        op_intensities = [
            ("Attn decode",  dc_s.attention_intensity()),
            ("FFN decode",   dc_s.ffn_intensity()),
            ("Attn prefill", pf.attention_intensity()),
            ("FFN prefill",  pf.ffn_intensity()),
        ]
        for label, intensity in op_intensities:
            op_color = OP_COLORS[label]
            fig.add_shape(
                type="line",
                x0=intensity, x1=intensity, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=op_color, width=1.5, dash=prec_dash),
            )
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                name=f"{label} ({prec_name}, {intensity:.1f})",
                line=dict(color=op_color, width=1.5, dash=prec_dash),
                legendgroup=f"ops_{prec_name}",
            ))

    _nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
             20000, 50000, 100000]
    x_min, x_max = x_range[0], x_range[-1]
    _xticks = [v for v in _nice if x_min <= v <= x_max * 1.1]
    _yticks = [v for v in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
               if y_min <= v <= y_max * 1.1]
    fig.update_layout(
        xaxis=dict(title="Arithmetic Intensity (FLOPs / byte)", type="log",
                   tickmode="array",
                   tickvals=_xticks,
                   ticktext=[str(v) for v in _xticks],
                   showgrid=True, gridcolor="#2a2a2a"),
        yaxis=dict(title="Achievable Peak Performance (TFLOP/s, est.)", type="log",
                   range=[np.log10(y_min), np.log10(y_max)],
                   tickmode="array",
                   tickvals=_yticks,
                   ticktext=[str(v) for v in _yticks],
                   showgrid=True, gridcolor="#2a2a2a"),
        legend=dict(orientation="v", x=1.01, y=1),
        template="plotly_dark",
        height=550,
        margin=dict(l=60, r=200, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Vertical lines show arithmetic intensity for each operation and phase. "
        "Where a line intersects a GPU's roofline = achieved performance for that operation. "
        "Source: Williams, Waterman, Patterson (2009) [WWP09]."
    )

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Pareto Frontier
# ─────────────────────────────────────────────────────────────────────────────

with tab_pareto:
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        pareto_x = st.radio("X-axis latency metric", ["ITL (ms)", "TTFT (ms)"], horizontal=True)
    with ctrl_col2:
        show_all = st.toggle("Show all configs", value=False)

    x_key = "itl_ms" if pareto_x == "ITL (ms)" else "ttft_ms"

    def _compute_pareto(points, lat_key):
        frontier = [
            p for p in points
            if not any(
                q["cost_per_m"] <= p["cost_per_m"] and
                q[lat_key]      <= p[lat_key] and
                q is not p
                for q in points
            )
        ]
        frontier.sort(key=lambda x: x[lat_key])
        return frontier

    fig_pareto = go.Figure()
    # Collect every pareto point so we can default-select the cheapest one
    # entries: (point_dict, gpu_obj, prec_name)
    _all_pareto = []

    with st.spinner("Sweeping parallelism configs…"):
        for idx, (gpu, color) in enumerate(zip(selected_gpus, colors)):
            short_name = gpu.name.replace("NVIDIA ", "").replace("AMD ", "")

            for prec_name in (selected_precisions or ["BF16"]):
                if prec_name == "FP4" and not gpu.supports_fp4:
                    continue

                p_dtype_bytes = DTYPE_OPTIONS[prec_name]["dtype_bytes"]
                p_compute_scale = DTYPE_OPTIONS[prec_name]["compute_scale"]
                dash = PRECISION_DASH[prec_name]
                trace_name = f"{short_name} ({prec_name})"
                legendgroup = f"{gpu.name}_{prec_name}"

                all_pts, pareto_itl, pareto_ttft = sweep_pareto_configs(
                    gpu.name, selected_model_name,
                    prefill_seq_len, decode_seq_len,
                    power_cost, pue,
                    p_dtype_bytes, p_compute_scale,
                    target_utilization,
                )

                # Utilization-adjusted cost: scale cost_per_m by realized utilization.
                # realized_util = typical_utilization × (tok_s_per_gpu / peak_tok_s_per_gpu)
                # where peak is the max throughput achievable at this (tp,ep,pp,num_gpus) config.
                # This reflects that small batches (tight ITL) leave the GPU mostly idle.
                from collections import defaultdict
                peak_by_cfg = defaultdict(float)
                for p in all_pts:
                    k = (p["tp"], p["ep"], p["pp"], p["num_gpus"])
                    peak_by_cfg[k] = max(peak_by_cfg[k], p["tok_s"] / max(1, p["num_gpus"]))

                adjusted_pts = []
                for p in all_pts:
                    k = (p["tp"], p["ep"], p["pp"], p["num_gpus"])
                    tok_s_per_gpu = p["tok_s"] / max(1, p["num_gpus"])
                    peak = peak_by_cfg[k]
                    throughput_frac = tok_s_per_gpu / peak if peak > 0 else 1.0
                    realized_util = target_utilization * throughput_frac
                    util_scale = target_utilization / realized_util if realized_util > 0 else 1.0
                    ap = dict(p)
                    ap["cost_per_m"] = p["cost_per_m"] * util_scale
                    ap["realized_util"] = realized_util
                    adjusted_pts.append(ap)

                pareto_pts = _compute_pareto(adjusted_pts, x_key)
                all_pts_display = adjusted_pts

                for p in pareto_pts:
                    _all_pareto.append((p, gpu, prec_name))

                if show_all:
                    dominated = [p for p in all_pts_display if p not in pareto_pts]
                    fig_pareto.add_trace(go.Scatter(
                        x=[p[x_key] for p in dominated],
                        y=[p["cost_per_m"] for p in dominated],
                        mode="markers",
                        name=trace_name,
                        legendgroup=legendgroup,
                        showlegend=False,
                        marker=dict(color=color, size=6, opacity=0.2),
                        customdata=[[p["tp"], p["ep"], p["pp"], p["num_gpus"], p["tok_s"], p["eff"], p["batch_size"], gpu.name, prec_name, p.get("realized_util", target_utilization)]
                                    for p in dominated],
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>"
                            "TP=%{customdata[0]}  EP=%{customdata[1]}  PP=%{customdata[2]}<br>"
                            "Batch: %{customdata[6]}  GPUs: %{customdata[3]}<br>"
                            f"{pareto_x}: %{{x:.2f}} ms<br>"
                            "$/M tokens: $%{y:.2f}<br>"
                            "Tok/s: %{customdata[4]:.0f}<br>"
                            "Parallelism eff.: %{customdata[5]:.1%}<extra></extra>"
                        ),
                    ))

                fig_pareto.add_trace(go.Scatter(
                    x=[p[x_key] for p in pareto_pts],
                    y=[p["cost_per_m"] for p in pareto_pts],
                    mode="markers+lines",
                    name=trace_name,
                    legendgroup=legendgroup,
                    showlegend=True,
                    marker=dict(color=color, size=7, opacity=1.0,
                                line=dict(width=1, color="white")),
                    line=dict(color=color, width=2, dash=dash),
                    customdata=[[p["tp"], p["ep"], p["pp"], p["num_gpus"], p["tok_s"], p["eff"], p["batch_size"], gpu.name, prec_name, p.get("realized_util", target_utilization)]
                                for p in pareto_pts],
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "TP=%{customdata[0]}  EP=%{customdata[1]}  PP=%{customdata[2]}<br>"
                        "Batch: %{customdata[6]}  GPUs: %{customdata[3]}<br>"
                        f"{pareto_x}: %{{x:.2f}} ms<br>"
                        "$/M tokens: $%{y:.2f}<br>"
                        "Tok/s: %{customdata[4]:.0f}<br>"
                        "Parallelism eff.: %{customdata[5]:.1%}<extra></extra>"
                    ),
                ))

    fig_pareto.update_layout(
        xaxis=dict(title=pareto_x, rangemode="nonnegative"),
        yaxis=dict(title="Cost per Million Output Tokens (USD)", rangemode="nonnegative"),
        template="plotly_dark",
        height=520,
        legend=dict(
            orientation="h", x=0.0, y=1.08,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=10),
            tracegroupgap=4,
        ),
        hovermode="closest",
    )

    chart_col, cost_col = st.columns([3, 2])

    with chart_col:
        selected = st.plotly_chart(
            fig_pareto, use_container_width=True,
            on_select="rerun", selection_mode="points", key="pareto_chart",
        )
        st.caption(
            "Pareto-optimal curves: no config is both cheaper AND faster. "
            "Click any point to inspect its details."
        )

    with cost_col:
        # Resolve which point to display
        pts = selected.selection.points if selected and selected.selection else []
        if pts:
            cd = pts[0]["customdata"]
            tp_sel, ep_sel, pp_sel, ngpus_sel, toks_sel, eff_sel, bs_sel, gpu_name_sel, prec_name_sel, realized_util_sel = (
                int(cd[0]), int(cd[1]), int(cd[2]), int(cd[3]),
                cd[4], cd[5], int(cd[6]), cd[7], cd[8], cd[9],
            )
            x_val = pts[0]["x"]
            y_val = pts[0]["y"]
            sel_gpu = gpu_name_map[gpu_name_sel]
        elif _all_pareto:
            # Default: cheapest Pareto-optimal point across all GPUs × precisions
            best_p, sel_gpu, prec_name_sel = min(_all_pareto, key=lambda t: t[0]["cost_per_m"])
            tp_sel    = int(best_p["tp"])
            ep_sel    = int(best_p["ep"])
            pp_sel    = int(best_p["pp"])
            ngpus_sel = int(best_p["num_gpus"])
            toks_sel  = best_p["tok_s"]
            eff_sel   = best_p["eff"]
            bs_sel    = int(best_p["batch_size"])
            realized_util_sel = best_p.get("realized_util", sel_gpu.typical_utilization)
            gpu_name_sel = sel_gpu.name
            x_val     = best_p[x_key]
            y_val     = best_p["cost_per_m"]
        else:
            sel_gpu = None
            prec_name_sel = "BF16"

        if sel_gpu is not None:
            from dataclasses import replace as dc_replace
            _prec = DTYPE_OPTIONS[prec_name_sel]
            _model_sel = dc_replace(model_name_map[selected_model_name],
                                    weight_dtype_bytes=_prec["dtype_bytes"],
                                    kv_cache_dtype_bytes=_prec["dtype_bytes"])
            short = sel_gpu.name.replace("NVIDIA ", "").replace("AMD ", "")
            st.markdown(f"**{short} ({prec_name_sel}) — TP{tp_sel} EP{ep_sel} PP{pp_sel} · Batch {bs_sel} · {ngpus_sel} GPUs**")

            m1, m2 = st.columns(2)
            m1.metric(pareto_x, f"{x_val:.2f} ms")
            m2.metric("$/M tokens", f"${y_val:.2f}")

            m3, m4 = st.columns(2)
            m3.metric("Parallelism eff.", f"{eff_sel:.1%}",
                      help="Fraction of theoretical throughput retained after tensor/pipeline parallelism communication overhead. Always 100% for single-GPU configs; drops with TP>1 due to AllReduce latency.")
            m4.metric("Cluster util.", f"{target_utilization:.0%}",
                      help="Fraction of time the cluster is actively serving requests, set by the slider above. Higher utilization spreads fixed CapEx over more tokens, lowering $/M tokens.")

            m5, m6 = st.columns(2)
            tok_s_per_gpu_sel = toks_sel / max(1, ngpus_sel)
            m5.metric("tok/s/GPU", f"{tok_s_per_gpu_sel:.0f}",
                      help="Decode throughput per GPU at this batch size and parallelism config. Use this to size your fleet: num_GPUs_needed = target_total_tok_s ÷ tok/s/GPU.")



# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "MoE Roofline Oracle · First-principles analytical simulator · "
    "All results are theoretical upper bounds, not empirical benchmarks. "
    "Roofline model: Williams, Waterman, Patterson (2009). "
    "Parallelism formulas: Narayanan et al. (2021), Lepikhin et al. (2021), "
    "Rajbhandari et al. (2022). "
    "Economics: EIA (2024), Uptime Institute (2023), SemiAnalysis (2024)."
)
