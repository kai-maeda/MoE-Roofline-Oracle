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
from oracle.workload import ALL_MODELS, ModelStats
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
}

COST_PROVIDERS = {
    "Hyperscaler":      {"power_cost": 0.04, "pue": 1.12, "desc": "Long-term PPA, campus-grade cooling (Google/MS/AWS)"},
    "Neocloud":         {"power_cost": 0.07, "pue": 1.25, "desc": "CoreWeave / Lambda / Crusoe tier"},
    "3yr Colo Rental":  {"power_cost": 0.10, "pue": 1.40, "desc": "Shared colocation facility"},
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_model_name = st.selectbox("Model", options=list(model_name_map.keys()), index=1)
with col2:
    preset = st.selectbox("Workload (ISL / OSL)", list(ISL_OSL_PRESETS.keys()), index=1)
with col3:
    dtype_name = st.selectbox("Precision", list(DTYPE_OPTIONS.keys()), index=0)
with col4:
    provider = st.selectbox("Cost provider", list(COST_PROVIDERS.keys()), index=1)

model = model_name_map[selected_model_name]
prefill_seq_len, decode_seq_len = ISL_OSL_PRESETS[preset]
dtype_bytes         = DTYPE_OPTIONS[dtype_name]["dtype_bytes"]
dtype_compute_scale = DTYPE_OPTIONS[dtype_name]["compute_scale"]
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

    tok_s_decode = profile.tokens_per_second_decode * decode_eff
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
        # Decode
        "decode_bottleneck":   profile.decode_result.bottleneck,
        "decode_mfu":          profile.decode_result.mfu_theoretical,
        "decode_intensity":    profile.decode_result.arithmetic_intensity,
        "itl_ms":              profile.inter_token_latency_ms,
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
                         dtype_bytes=2.0, dtype_compute_scale=1.0):
    from dataclasses import replace as dc_replace
    gpu   = gpu_name_map[gpu_name]
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

    all_points = []
    for cfg in configs:
        num_gpus  = cfg.total_gpus
        num_nodes = max(1, num_gpus // gpus_per_node)
        cluster   = ClusterConfig(gpu=gpu, fabric=fab, num_nodes=num_nodes, gpus_per_node=gpus_per_node)
        capex     = compute_capex(cluster)
        tco       = TCOResult(cluster=cluster, capex=capex, assumptions=eco)

        for batch_size in batch_sizes:
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

            decode_compute_s  = (profile.inter_token_latency_ms  / 1e3) / decode_speedup
            prefill_compute_s = (profile.time_to_first_token_ms  / 1e3) / prefill_speedup
            batch_compute_s   = profile.decode_result.bottleneck_time_s / decode_speedup

            comm_s = overhead.total_comm_time_s

            itl_ms  = (decode_compute_s  + comm_s) * 1e3
            ttft_ms = (prefill_compute_s + comm_s) * 1e3

            batch_step_s = batch_compute_s + comm_s
            tok_s        = batch_size / batch_step_s if batch_step_s > 0 else 0.0

            eff        = decode_compute_s / (decode_compute_s + comm_s) if (decode_compute_s + comm_s) > 0 else 1.0
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


results = []
with st.spinner("Running analysis…"):
    for gpu in selected_gpus:
        r = run_analysis(
            gpu.name, selected_model_name,
            prefill_seq_len, decode_seq_len,
            8, 1, 1, 1, power_cost, pue,
            dtype_bytes, dtype_compute_scale,
        )
        results.append(r)


# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────

colors = px.colors.qualitative.Plotly

tab_pareto, tab_roofline = st.tabs([
    "🎯 Pareto Frontier",
    "📈 Roofline Chart",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Roofline Chart
# ─────────────────────────────────────────────────────────────────────────────

with tab_roofline:
    st.subheader("Roofline Chart")

    with st.expander("Hardware specs reference"):
        spec_rows = []
        for g in selected_gpus:
            spec_rows.append({
                "GPU": g.name,
                "Arch": g.architecture,
                "BF16 TFLOPS": f"{g.flops_bf16:.0f}",
                "HBM BW (TB/s)": f"{g.hbm_bandwidth_tb:.1f}",
                "HBM Cap (GB)": f"{g.hbm_capacity_gb:.0f}",
                "Ridge (FLOPs/B)": f"{g.arithmetic_intensity_ridge:.0f}",
                "TDP (W)": f"{g.tdp_w:.0f}",
                "$/TFLOP": f"${g.cost_per_tflop:.1f}",
                "$/TB/s": f"${g.cost_per_tb_bandwidth:.0f}",
                "Confirmed": "✓" if g.specs_confirmed else "~ est.",
            })
        st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

    fig = go.Figure()

    # x range: 0.1 → 10 000 FLOPs/byte covers all realistic MoE workloads
    x_range = np.logspace(-1, 4, 1000)

    # y range: 1 TFLOP/s floor → 20% above the highest GPU peak ceiling
    max_peak = max(r["_gpu_obj"].flops_bf16 * dtype_compute_scale for r in results)
    y_min, y_max = 1.0, max_peak * 1.2

    for idx, (r, color) in enumerate(zip(results, colors)):
        gpu  = r["_gpu_obj"]
        peak_compute = gpu.flops_bf16 * dtype_compute_scale
        peak_bw      = gpu.hbm_bandwidth_tb

        y_roof = np.minimum(peak_compute, peak_bw * x_range)

        fig.add_trace(go.Scatter(
            x=x_range, y=y_roof,
            mode="lines",
            name=gpu.name,
            line=dict(color=color, width=2, dash="solid"),
            legendgroup=gpu.name,
            showlegend=True,
        ))

    # Vertical lines for per-operation intensities (workload-only, same for all GPUs)
    from dataclasses import replace as dc_replace
    _model_dtype = dc_replace(model, weight_dtype_bytes=dtype_bytes, kv_cache_dtype_bytes=dtype_bytes)
    prefill_stats = ModelStats(model=_model_dtype, seq_len=prefill_seq_len, batch_size=8, phase="prefill")
    decode_stats  = ModelStats(model=_model_dtype, seq_len=decode_seq_len,  batch_size=8, phase="decode")
    op_intensities = [
        ("Attn decode",  decode_stats.attention_intensity(),  "#F0E68C", "dot"),
        ("FFN decode",   decode_stats.ffn_intensity(),        "#FFA07A", "dot"),
        ("Attn prefill", prefill_stats.attention_intensity(), "#00CED1", "solid"),
        ("FFN prefill",  prefill_stats.ffn_intensity(),       "#98FB98", "solid"),
    ]

    for label, intensity, color, dash in op_intensities:
        # add_shape with yref="paper" spans full plot height regardless of zoom
        fig.add_shape(
            type="line",
            x0=intensity, x1=intensity,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=color, width=1.5, dash=dash),
        )
        # Dummy trace for legend entry only
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name=f"{label} ({intensity:.1f})",
            line=dict(color=color, width=1.5, dash=dash),
            legendgroup="ops",
        ))

    _nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
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
        yaxis=dict(title="Achieved Performance (TFLOP/s)", type="log",
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
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        pareto_x = st.radio("X-axis latency metric", ["ITL (ms)", "TTFT (ms)"], horizontal=True)
    with ctrl_col2:
        show_all = st.toggle("Show all configs", value=False)

    x_key = "itl_ms" if pareto_x == "ITL (ms)" else "ttft_ms"

    fig_pareto = go.Figure()
    # Collect every pareto point so we can default-select the cheapest one
    _all_pareto: list[tuple[dict, object]] = []  # (point_dict, gpu_obj)

    with st.spinner("Sweeping parallelism configs…"):
        for idx, (r, color) in enumerate(zip(results, colors)):
            gpu = r["_gpu_obj"]
            short_name = gpu.name.replace("NVIDIA ", "").replace("AMD ", "")

            all_pts, pareto_itl, pareto_ttft = sweep_pareto_configs(
                gpu.name, selected_model_name,
                prefill_seq_len, decode_seq_len,
                power_cost, pue,
                dtype_bytes, dtype_compute_scale,
            )
            pareto_pts = pareto_itl if x_key == "itl_ms" else pareto_ttft

            for p in pareto_pts:
                _all_pareto.append((p, gpu))

            if show_all:
                dominated = [p for p in all_pts if p not in pareto_pts]
                fig_pareto.add_trace(go.Scatter(
                    x=[p[x_key] for p in dominated],
                    y=[p["cost_per_m"] for p in dominated],
                    mode="markers",
                    name=short_name,
                    legendgroup=gpu.name,
                    showlegend=False,
                    marker=dict(color=color, size=6, opacity=0.2),
                    customdata=[[p["tp"], p["ep"], p["pp"], p["num_gpus"], p["tok_s"], p["eff"], p["batch_size"], gpu.name]
                                for p in dominated],
                    hovertemplate=(
                        f"<b>{short_name}</b><br>"
                        "TP=%{customdata[0]}  EP=%{customdata[1]}  PP=%{customdata[2]}<br>"
                        "Batch: %{customdata[6]}  GPUs: %{customdata[3]}<br>"
                        f"{pareto_x}: %{{x:.2f}} ms<br>"
                        "$/M tokens: $%{y:.2f}<br>"
                        "Tok/s: %{customdata[4]:.0f}<br>"
                        "Efficiency: %{customdata[5]:.1%}<extra></extra>"
                    ),
                ))

            fig_pareto.add_trace(go.Scatter(
                x=[p[x_key] for p in pareto_pts],
                y=[p["cost_per_m"] for p in pareto_pts],
                mode="markers+lines",
                name=short_name,
                legendgroup=gpu.name,
                showlegend=True,
                marker=dict(color=color, size=7, opacity=1.0,
                            line=dict(width=1, color="white")),
                line=dict(color=color, width=2),
                customdata=[[p["tp"], p["ep"], p["pp"], p["num_gpus"], p["tok_s"], p["eff"], p["batch_size"], gpu.name]
                            for p in pareto_pts],
                hovertemplate=(
                    f"<b>{short_name}</b><br>"
                    "TP=%{customdata[0]}  EP=%{customdata[1]}  PP=%{customdata[2]}<br>"
                    "Batch: %{customdata[6]}  GPUs: %{customdata[3]}<br>"
                    f"{pareto_x}: %{{x:.2f}} ms<br>"
                    "$/M tokens: $%{y:.2f}<br>"
                    "Tok/s: %{customdata[4]:.0f}<br>"
                    "Efficiency: %{customdata[5]:.1%}<extra></extra>"
                ),
            ))

    fig_pareto.update_layout(
        xaxis=dict(title=pareto_x, rangemode="nonnegative"),
        yaxis=dict(title="Cost per Million Output Tokens (USD)", rangemode="nonnegative"),
        template="plotly_dark",
        height=520,
        legend=dict(
            orientation="v", x=0.99, y=0.99,
            xanchor="right", yanchor="top",
            bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1,
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
            "Click any point to inspect its cost breakdown."
        )

    with cost_col:
        # Resolve which point to display
        pts = selected.selection.points if selected and selected.selection else []
        if pts:
            cd = pts[0]["customdata"]
            tp_sel, ep_sel, pp_sel, ngpus_sel, toks_sel, eff_sel, bs_sel, gpu_name_sel = (
                int(cd[0]), int(cd[1]), int(cd[2]), int(cd[3]),
                cd[4], cd[5], int(cd[6]), cd[7],
            )
            x_val = pts[0]["x"]
            y_val = pts[0]["y"]
            sel_gpu = gpu_name_map[gpu_name_sel]
        elif _all_pareto:
            # Default: cheapest Pareto-optimal point across all GPUs
            best_p, sel_gpu = min(_all_pareto, key=lambda t: t[0]["cost_per_m"])
            tp_sel    = int(best_p["tp"])
            ep_sel    = int(best_p["ep"])
            pp_sel    = int(best_p["pp"])
            ngpus_sel = int(best_p["num_gpus"])
            toks_sel  = best_p["tok_s"]
            eff_sel   = best_p["eff"]
            bs_sel    = int(best_p["batch_size"])
            gpu_name_sel = sel_gpu.name
            x_val     = best_p[x_key]
            y_val     = best_p["cost_per_m"]
        else:
            sel_gpu = None

        if sel_gpu is not None:
            gpus_per_node_sel = 72 if "NVL72" in sel_gpu.name else 8
            num_nodes_sel = max(1, ngpus_sel // gpus_per_node_sel)
            sel_cluster = ClusterConfig(gpu=sel_gpu, fabric=IB_NDR,
                                        num_nodes=num_nodes_sel, gpus_per_node=gpus_per_node_sel)
            sel_capex = compute_capex(sel_cluster)
            sel_tco   = TCOResult(cluster=sel_cluster, capex=sel_capex,
                                  assumptions=EconomicAssumptions(power_cost_per_kwh=power_cost, pue=pue))
            sel_profile = InferenceProfile(
                model=model_name_map[selected_model_name], gpu=sel_gpu,
                prefill_seq_len=prefill_seq_len, prefill_batch_size=bs_sel,
                decode_seq_len=decode_seq_len,   decode_batch_size=bs_sel,
            )

            short = sel_gpu.name.replace("NVIDIA ", "").replace("AMD ", "")
            st.markdown(f"**{short} — TP{tp_sel} EP{ep_sel} PP{pp_sel} · Batch {bs_sel} · {ngpus_sel} GPUs**")

            m1, m2 = st.columns(2)
            m1.metric(pareto_x, f"{x_val:.2f} ms")
            m2.metric("$/M tokens", f"${y_val:.2f}")

            m3, m4 = st.columns(2)
            m3.metric("Tok/s", f"{toks_sel:.0f}")
            m4.metric("Parallelism eff.", f"{eff_sel:.1%}")

            m5, m6 = st.columns(2)
            m5.metric("Prefill MFU", f"{sel_profile.prefill_result.mfu_theoretical:.1%}")
            m6.metric("Decode MFU",  f"{sel_profile.decode_result.mfu_theoretical:.1%}")

            st.divider()
            st.markdown("**CapEx**")
            e1, e2 = st.columns(2)
            e1.metric("GPU", f"${sel_capex.gpu_cost_usd:,.0f}")
            e2.metric("Server", f"${sel_capex.server_cost_usd:,.0f}")
            e3, e4 = st.columns(2)
            e3.metric("Network", f"${sel_capex.network_cost_usd:,.0f}")
            e4.metric("Total CapEx", f"${sel_capex.total_usd:,.0f}")

            st.markdown("**TCO**")
            t1, t2 = st.columns(2)
            t1.metric("Annual Power", f"${sel_tco.annual_power_cost_usd:,.0f}")
            t2.metric("$/GPU-hr", f"${sel_tco.cost_per_gpu_hour_usd:.2f}")
            t3, t4 = st.columns(2)
            t3.metric("Annual TCO", f"${sel_tco.annual_tco_usd:,.0f}")
            t4.metric(f"Total TCO ({sel_gpu.depreciation_years:.0f}yr)", f"${sel_tco.total_tco_usd:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────

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
