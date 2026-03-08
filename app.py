"""
╔══════════════════════════════════════════════════════════════════╗
║         AI WATER USAGE EXPLORER — Streamlit Dashboard           ║
║         How thirsty is AI, really?                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Purpose: Educate the general public about the hidden water     ║
║  footprint of AI systems across four key stages:               ║
║    1. Inference  — water used per prompt                        ║
║    2. Training   — one-time cost to build the model             ║
║    3. Development— fine-tuning, RLHF, safety testing            ║
║    4. Electricity— water used to generate the power AI runs on  ║
║                                                                 ║
║  Primary Source: Li et al. (2023) "Making AI Less Thirsty"      ║
║  NOTE: Most AI companies do NOT disclose water usage data.      ║
║  All undisclosed values are derived estimates.                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be the very first Streamlit call
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Water Usage Explorer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS
# Design philosophy: deep-ocean dark theme with electric-blue/cyan
# accents — evoking both water and technology. Clean, editorial feel
# with strong typographic hierarchy.
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  /* ── Global reset & base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── App background ── */
  .stApp {
    background: linear-gradient(160deg, #060c1a 0%, #0a1628 50%, #061020 100%);
    color: #e2eaf5;
  }

  /* ── Main content area ── */
  .block-container {
    padding-top: 2rem;
    max-width: 1400px;
  }

  /* ── Headings ── */
  h1 {
    font-family: 'Space Mono', monospace !important;
    color: #38bdf8 !important;
    font-size: 2.6rem !important;
    letter-spacing: -1px;
  }
  h2 {
    font-family: 'Space Mono', monospace !important;
    color: #7dd3fc !important;
    font-size: 1.5rem !important;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
  }
  h3 {
    color: #93c5fd !important;
    font-weight: 600 !important;
  }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2540 0%, #0c1e36 100%);
    border: 1px solid #1e4976;
    border-radius: 12px;
    padding: 1rem !important;
    box-shadow: 0 4px 24px rgba(56, 189, 248, 0.08);
  }
  [data-testid="metric-container"] label {
    color: #7dd3fc !important;
    font-size: 0.8rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101e 0%, #081526 100%) !important;
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label {
    color: #94a3b8 !important;
  }
  [data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Assumption boxes ── */
  .assumption-box {
    background: rgba(14, 42, 80, 0.6);
    border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
  }
  .assumption-box b { color: #38bdf8; }

  /* ── Warning/disclosure boxes ── */
  .disclosure-box {
    background: rgba(234, 179, 8, 0.08);
    border-left: 3px solid #eab308;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.85rem;
    color: #a3a37a;
  }
  .disclosure-box b { color: #fcd34d; }

  /* ── Real-world comparison cards ── */
  .comparison-card {
    background: linear-gradient(135deg, #0f2540 0%, #0c1e36 100%);
    border: 1px solid #1e4976;
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
    margin: 6px 2px;
    transition: border-color 0.2s;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
  }
  .comparison-card:hover { border-color: #38bdf8; }
  .comparison-card .icon   { font-size: 2.4rem; line-height: 1.2; }
  .comparison-card .count  { font-family: 'Space Mono', monospace; font-size: 1.7rem;
                              color: #38bdf8; font-weight: 700; margin: 4px 0; }
  .comparison-card .label  { font-size: 0.82rem; color: #64748b; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #060c1a;
    border-bottom: 1px solid #1e3a5f;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    background: #0f2540 !important;
    border-bottom: 2px solid #38bdf8 !important;
  }

  /* ── Sliders / inputs ── */
  [data-testid="stSlider"] > div > div > div {
    background: #38bdf8 !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #0d1f35 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #7dd3fc !important;
    font-weight: 600 !important;
  }

  /* ── Footer ── */
  .footer-text {
    text-align: center;
    color: #334155;
    font-size: 0.8rem;
    padding: 24px 0 12px;
    border-top: 1px solid #1e3a5f;
    margin-top: 32px;
    line-height: 1.8;
  }
  .footer-text a { color: #38bdf8; text-decoration: none; }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 1: CONSTANTS & CONVERSION FACTORS
# ══════════════════════════════════════════════════════════════════

# Standard unit conversion
LITERS_TO_GALLONS  = 0.264172   # 1 liter  = 0.264172 US gallons
ML_TO_GALLONS      = 0.000264172  # 1 mL = 0.000264172 US gallons
GALLONS_TO_LITERS  = 3.78541   # 1 gallon = 3.78541 liters

# Real-world water benchmarks (all in gallons)
# Sources: EPA WaterSense program; standard US household measurements
WATER_BENCHMARKS = {
    ("🥛", "8 oz drinking glasses"):      0.0625,    # 8 fl oz
    ("🍶", "16.9 oz water bottles"):       0.1320,    # standard 500mL bottle
    ("☕", "cups of coffee (brewed)"):     0.0313,    # ~4 oz water to brew
    ("🚽", "toilet flushes"):              1.6,       # EPA WaterSense standard
    ("🚿", "minutes in the shower"):       2.0,       # EPA avg: ~2 gal/min
    ("🛁", "standard bathtubs"):           36.0,      # US average fill
    ("🏊", "Olympic swimming pools"):      660_000.0, # 2.5M liters
}

# Electricity water intensity (US average, per USGS)
# "Withdrawn"  = all water pulled from source (most returned)
# "Consumed"   = water NOT returned (evaporated, etc.)
ELEC_WATER_WITHDRAWN_GAL_KWH  = 1.8    # USGS National Water-Use Science Project
ELEC_WATER_CONSUMED_GAL_KWH   = 0.3    # USGS ibid.


# ══════════════════════════════════════════════════════════════════
# SECTION 2: AI SYSTEM REFERENCE DATA
#
# ⚠️  TRANSPARENCY NOTE:
#   The vast majority of AI companies do NOT publicly disclose
#   energy consumption or water usage at the model or prompt level.
#   Values below are derived from:
#     (a) Peer-reviewed estimates (Li et al. 2023)
#     (b) Published corporate sustainability/environmental reports
#     (c) Researcher estimates in published ML literature
#   They represent PLAUSIBLE ORDER-OF-MAGNITUDE figures,
#   NOT precise or officially-verified measurements.
# ══════════════════════════════════════════════════════════════════

AI_SYSTEMS = {

    # ── ChatGPT / GPT-4 ────────────────────────────────────────
    # Hosted by OpenAI on Microsoft Azure infrastructure.
    "ChatGPT (GPT-4)": {
        "company":   "OpenAI / Microsoft Azure",
        "color":     "#10b981",   # teal-green
        "disclosed": False,       # OpenAI has NOT publicly disclosed water figures

        # Inference water (direct cooling, mL per prompt):
        # Li et al. (2023) estimate ~500mL per 10–50 prompts for ChatGPT
        # at a US data center.  Mid-point: 500mL / 20 = 25mL.
        # We use 25mL as the baseline; user can override in sidebar.
        "inference_ml_per_prompt":  25.0,

        # Training water (total liters, one-time):
        # Li et al. (2023) estimate GPT-3 training used ~700,000 L.
        # GPT-4 is widely reported as significantly larger (not disclosed).
        # We apply a conservative 5× multiplier → ~3.5M L.
        # ⚠️ ESTIMATE ONLY — no official figure exists.
        "training_liters":          3_500_000,

        # Development fraction:
        # RLHF, fine-tuning, safety evaluations, and ablation studies
        # typically consume 20–40% of full training cost in practice.
        # Using 30% as a reasonable midpoint based on ML engineering lit.
        "dev_fraction_of_training": 0.30,

        # Data center WUE (Water Usage Effectiveness, liters per kWh):
        # Microsoft 2022 Environmental Sustainability Report: WUE ≈ 0.49 L/kWh
        # (global average across Azure fleet)
        "data_center_wue":          0.49,

        # Power per prompt (kWh):
        # GPU inference benchmarks suggest 0.001–0.01 kWh per query.
        # OpenAI has not disclosed this; using 0.003 kWh as a mid estimate
        # consistent with literature on GPT-class model inference.
        "power_per_prompt_kwh":     0.003,
    },

    # ── Claude (Anthropic) ─────────────────────────────────────
    # Hosted on AWS and GCP; Constitutional AI training approach.
    "Claude (Anthropic)": {
        "company":   "Anthropic / AWS + GCP",
        "color":     "#f97316",   # orange
        "disclosed": False,       # Anthropic has NOT publicly disclosed water figures

        # Inference estimate: Comparable to GPT-4 class architecturally.
        # Slightly lower estimate based on Constitutional AI potentially
        # requiring fewer inference-time operations. ⚠️ ESTIMATE ONLY.
        "inference_ml_per_prompt":  20.0,

        # Training estimate: Comparable scale to GPT-4; no disclosure.
        # Using slightly lower value to reflect Anthropic's reported
        # focus on training efficiency. ⚠️ ESTIMATE ONLY.
        "training_liters":          2_800_000,

        "dev_fraction_of_training": 0.25,

        # Claude runs on both AWS and GCP; using weighted average WUE.
        # AWS 2022 Sustainability Report: WUE ≈ 0.18 L/kWh
        # Google 2022 Environmental Report: WUE ≈ 1.0 L/kWh
        # Assumed 60/40 AWS/GCP split → weighted avg ≈ 0.51 L/kWh
        "data_center_wue":          0.51,

        "power_per_prompt_kwh":     0.0025,
    },

    # ── Gemini (Google DeepMind) ───────────────────────────────
    # Hosted natively on Google Cloud Platform (GCP).
    "Gemini (Google)": {
        "company":   "Google DeepMind / GCP",
        "color":     "#4285f4",   # Google blue
        "disclosed": False,       # Google has NOT disclosed model-specific water data

        # Inference estimate: Google has hinted Gemini Ultra is comparable
        # to GPT-4 in scale. Slightly higher direct water estimate because
        # GCP has a higher WUE (1.0 L/kWh vs Azure's 0.49). ⚠️ ESTIMATE.
        "inference_ml_per_prompt":  28.0,

        # Training estimate: Gemini Ultra assumed comparable to GPT-4.
        # Google has not disclosed this figure. ⚠️ ESTIMATE ONLY.
        "training_liters":          4_000_000,

        "dev_fraction_of_training": 0.28,

        # Google 2022 Environmental Report: GCP WUE ≈ 1.0 L/kWh
        "data_center_wue":          1.0,

        "power_per_prompt_kwh":     0.003,
    },

    # ── Llama 3 (Meta) ─────────────────────────────────────────
    # Open-source model; inference often on own or cloud hardware.
    # Using Meta data center metrics for hosted Llama (Meta AI app).
    "Llama 3 (Meta)": {
        "company":   "Meta AI",
        "color":     "#a855f7",   # purple
        "disclosed": False,       # Meta has NOT disclosed model-specific water data

        # Inference estimate: Llama 3 70B is a smaller model than GPT-4
        # class systems, resulting in lower per-prompt compute & cooling.
        # ⚠️ ESTIMATE ONLY.
        "inference_ml_per_prompt":  12.0,

        # Training estimate: Llama 3 70B is considerably smaller than GPT-4.
        # Meta has not disclosed exact training water. ⚠️ ESTIMATE ONLY.
        "training_liters":          1_200_000,

        "dev_fraction_of_training": 0.20,

        # Meta 2022 Sustainability Report: data center WUE ≈ 0.26 L/kWh
        # (Meta is among the more water-efficient data center operators)
        "data_center_wue":          0.26,

        "power_per_prompt_kwh":     0.002,
    },
}

# Sorted model list for consistent ordering in charts
MODEL_NAMES = list(AI_SYSTEMS.keys())


# ══════════════════════════════════════════════════════════════════
# SECTION 3: SIDEBAR — User Settings & Assumption Overrides
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💧 AI Water Explorer")
    st.caption("Adjust settings to explore how assumptions affect water estimates.")

    st.markdown("---")
    st.markdown("### 📐 Display Options")

    unit = st.radio(
        "Water unit",
        ["Gallons", "Liters", "Milliliters"],
        index=0,
        help="All calculations are done in gallons internally; this setting converts the display."
    )

    show_elec_water = st.checkbox(
        "Include electricity-related water",
        value=True,
        help=(
            "AI data centers use electricity. Generating that electricity "
            "also requires water (for thermoelectric cooling in power plants)."
        ),
    )

    elec_method = st.radio(
        "Electricity water method",
        ["Withdrawn (total flow)", "Consumed (net loss)"],
        help=(
            "Withdrawn: all water extracted from a source for cooling, "
            "most of which is returned. Consumed: water not returned "
            "(lost to evaporation, etc.). Consumed is the more conservative figure."
        ),
    )

    st.markdown("---")
    st.markdown("### 🎛️ Your Usage")

    daily_prompts = st.slider(
        "Your daily AI prompts",
        min_value=1, max_value=300, value=20,
        help="How many prompts/messages do you send to AI tools per day?"
    )

    amortization_years = st.slider(
        "Amortize training over (years)",
        min_value=1, max_value=10, value=3,
        help=(
            "Training is a one-time cost. This spreads that cost over a "
            "number of years to estimate a 'per day' contribution."
        ),
    )

    st.markdown("---")
    st.markdown("### 🔬 Override Assumptions")
    st.caption("Change per-model inference estimates (mL per prompt) below.")

    # Allow per-model inference overrides; default = values from AI_SYSTEMS dict
    custom_inference_ml = {}
    for m in MODEL_NAMES:
        default_val = float(AI_SYSTEMS[m]["inference_ml_per_prompt"])
        custom_inference_ml[m] = st.number_input(
            label=m.split("(")[0].strip(),
            min_value=0.1,
            max_value=300.0,
            value=default_val,
            step=0.5,
            key=f"inf_{m}",
        )

    elec_intensity = st.slider(
        "Electricity water intensity (gal / kWh)",
        min_value=0.05,
        max_value=4.0,
        value=(
            ELEC_WATER_WITHDRAWN_GAL_KWH
            if "Withdrawn" in elec_method
            else ELEC_WATER_CONSUMED_GAL_KWH
        ),
        step=0.05,
        help=(
            "USGS estimates: ~1.8 gal/kWh withdrawn, ~0.3 gal/kWh consumed "
            "for the US average electricity mix."
        ),
    )

    st.markdown("---")
    st.caption(
        "📖 Sources: Li et al. (2023); Microsoft, Google, Amazon, Meta "
        "Sustainability Reports (2022); USGS Water-Use Science Project."
    )


# ══════════════════════════════════════════════════════════════════
# SECTION 4: HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def to_display_unit(gallons: float) -> tuple[float, str]:
    """Convert a gallon value to the user-selected display unit."""
    if unit == "Liters":
        return gallons * GALLONS_TO_LITERS, "L"
    elif unit == "Milliliters":
        return gallons * GALLONS_TO_LITERS * 1000, "mL"
    return gallons, "gal"


def fmt_water(gallons: float) -> str:
    """Return a nicely-formatted string for a water volume in the display unit."""
    val, u = to_display_unit(gallons)
    if   val == 0:           return f"0 {u}"
    elif val < 0.001:        return f"{val:.6f} {u}"
    elif val < 0.1:          return f"{val:.4f} {u}"
    elif val < 1:            return f"{val:.3f} {u}"
    elif val < 1_000:        return f"{val:.2f} {u}"
    elif val < 1_000_000:    return f"{val/1_000:.1f}K {u}"
    else:                    return f"{val/1_000_000:.2f}M {u}"


def compute_water(model: str) -> dict:
    """
    Compute all water usage components for one AI model.

    Returns a dict with water quantities in gallons for:
      - inference_per_prompt_gal  : direct cooling water, per single prompt
      - inference_daily_gal       : inference water for [daily_prompts] prompts
      - elec_per_prompt_gal       : electricity water, per single prompt
      - elec_daily_gal            : electricity water for [daily_prompts] prompts
      - training_total_gal        : full one-time training water
      - dev_total_gal             : full one-time development/fine-tuning water
      - daily_total_gal           : inference + electricity water for one day
    """
    d = AI_SYSTEMS[model]

    # ── Inference (direct cooling) ──────────────────────────────
    # User-overridden mL value → convert to gallons
    inf_gal_per_prompt = custom_inference_ml[model] * ML_TO_GALLONS
    inf_daily_gal      = inf_gal_per_prompt * daily_prompts

    # ── Electricity water ───────────────────────────────────────
    # Split into:
    #   a) On-site data center cooling: power_per_prompt × WUE (L/kWh) → gallons
    #   b) Off-site power generation:   power_per_prompt × elec_intensity (gal/kWh)
    dc_cooling_gal_per_prompt = (
        d["power_per_prompt_kwh"] * d["data_center_wue"] * LITERS_TO_GALLONS
    )
    offsite_elec_gal_per_prompt = (
        d["power_per_prompt_kwh"] * elec_intensity if show_elec_water else 0.0
    )
    elec_per_prompt_gal = dc_cooling_gal_per_prompt + offsite_elec_gal_per_prompt
    elec_daily_gal      = elec_per_prompt_gal * daily_prompts

    # ── Training (one-time total) ───────────────────────────────
    training_gal_total = d["training_liters"] * LITERS_TO_GALLONS

    # ── Development (fraction of training) ─────────────────────
    dev_gal_total = training_gal_total * d["dev_fraction_of_training"]

    return {
        "inference_per_prompt_gal":  inf_gal_per_prompt,
        "inference_daily_gal":       inf_daily_gal,
        "elec_per_prompt_gal":       elec_per_prompt_gal,
        "elec_daily_gal":            elec_daily_gal,
        "training_total_gal":        training_gal_total,
        "dev_total_gal":             dev_gal_total,
        "daily_total_gal":           inf_daily_gal + elec_daily_gal,
    }


# Pre-compute water data for all models once (used across tabs)
water = {m: compute_water(m) for m in MODEL_NAMES}

# Plotly chart theme — consistent with the dark dashboard aesthetic
PLOTLY_TEMPLATE = "plotly_dark"
PLOT_BG   = "rgba(6, 12, 26, 0)"    # transparent over page background
PAPER_BG  = "rgba(6, 12, 26, 0)"
GRID_COL  = "#1e3a5f"
MODEL_COLORS = [AI_SYSTEMS[m]["color"] for m in MODEL_NAMES]


# ══════════════════════════════════════════════════════════════════
# SECTION 5: HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("# 💧 How Thirsty Is AI?")
st.markdown(
    "Explore the **hidden water footprint** of popular AI systems — "
    "from the moment you hit Send, to the massive resources required to build them."
)

# ── Global disclosure banner ────────────────────────────────────
st.markdown("""
<div class="disclosure-box">
  <b>⚠️ Transparency Notice:</b>
  OpenAI, Anthropic, Google, and Meta <b>do not publicly disclose</b> model-level
  water or energy usage data. All figures in this dashboard are <b>derived estimates</b>
  based on published academic research (Li et al. 2023), corporate sustainability
  reports, and known model architectures. Treat them as <em>informed order-of-magnitude
  approximations</em>, not official measurements. Methodology is detailed in each tab.
</div>
""", unsafe_allow_html=True)

# ── About expander ──────────────────────────────────────────────
with st.expander("ℹ️ How this works — methodology & sources", expanded=False):
    st.markdown("""
    ### The Four Stages of AI Water Use

    | Stage | What happens | Water source |
    |---|---|---|
    | **Inference** | GPUs process your prompt | Cooling towers at data center |
    | **Training** | Model learned from billions of examples | Cooling during multi-week GPU runs |
    | **Development** | Fine-tuning, safety testing, RLHF | Same as training, repeated many times |
    | **Electricity** | Power plants generate electricity AI runs on | Thermoelectric cooling at power plants |

    ### Key Sources
    - **Li, P. et al. (2023).** *"Making AI Less Thirsty: Uncovering and Addressing the Secret
      Water Footprint of AI Models."* — Primary source for inference and GPT-3 training estimates.
    - **Microsoft Environmental Sustainability Report 2022** — Azure WUE (0.49 L/kWh)
    - **Google Environmental Report 2022** — GCP WUE (1.0 L/kWh)
    - **Amazon Sustainability Report 2022** — AWS WUE (0.18 L/kWh)
    - **Meta Sustainability Report 2022** — Meta data center WUE (0.26 L/kWh)
    - **USGS National Water-Use Science Project** — Electricity water intensity

    ### WUE Explained
    *Water Usage Effectiveness* (WUE) measures how much water a data center uses per
    kilowatt-hour of IT load. Lower = more water-efficient.
    """)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: TOP-LINE KPI METRICS
# ══════════════════════════════════════════════════════════════════

st.markdown("### At a Glance — Water Per Prompt")
kpi_cols = st.columns(len(MODEL_NAMES))

for col, model in zip(kpi_cols, MODEL_NAMES):
    with col:
        val_str = fmt_water(water[model]["inference_per_prompt_gal"])
        co = AI_SYSTEMS[model]["company"].split("/")[0].strip()
        st.metric(
            label=model.split("(")[0].strip(),
            value=val_str,
            delta=co,
            delta_color="off",
        )

st.caption(
    f"Direct cooling water only. Using {daily_prompts} daily prompt assumption. "
    "Toggle electricity water in the sidebar."
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: TABS
# ══════════════════════════════════════════════════════════════════

tab_overview, tab_prompt, tab_training, tab_electricity, tab_realworld = st.tabs([
    "📊 Overview",
    "💬 Per Prompt",
    "🏗️ Training & Dev",
    "⚡ Electricity",
    "🌍 Real-World Scale",
])


# ────────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ────────────────────────────────────────────────────────────────
with tab_overview:
    st.header("Daily Water Footprint Overview")
    st.caption(
        f"Based on {daily_prompts} prompts/day across all four stages. "
        "Training cost is amortized over the selected period."
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Stacked bar: daily inference + electricity water per model
        inf_daily  = [to_display_unit(water[m]["inference_daily_gal"])[0]  for m in MODEL_NAMES]
        elec_daily = [to_display_unit(water[m]["elec_daily_gal"])[0]        for m in MODEL_NAMES]
        _, u_label = to_display_unit(1)

        fig_overview = go.Figure()
        fig_overview.add_trace(go.Bar(
            name="Inference (direct cooling)",
            x=MODEL_NAMES, y=inf_daily,
            marker_color="#38bdf8",
            hovertemplate="%{y:.5f} " + u_label + "<extra>Inference</extra>",
        ))
        fig_overview.add_trace(go.Bar(
            name="Electricity (power generation + DC cooling)",
            x=MODEL_NAMES, y=elec_daily,
            marker_color="#0369a1",
            hovertemplate="%{y:.5f} " + u_label + "<extra>Electricity</extra>",
        ))
        fig_overview.update_layout(
            barmode="stack",
            title=f"Daily Operational Water Use ({daily_prompts} prompts/day)",
            yaxis_title=f"Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=80),
        )
        st.plotly_chart(fig_overview, use_container_width=True)

    with col_right:
        # Radar / spider chart comparing the four dimensions
        categories = ["Inference", "Electricity", "Training (÷10K)", "Dev (÷10K)"]

        fig_radar = go.Figure()
        for model in MODEL_NAMES:
            w = water[model]
            # Normalise training/dev to be chart-comparable with daily figures
            vals = [
                to_display_unit(w["inference_daily_gal"])[0],
                to_display_unit(w["elec_daily_gal"])[0],
                to_display_unit(w["training_total_gal"] / 10_000)[0],
                to_display_unit(w["dev_total_gal"] / 10_000)[0],
            ]
            vals += [vals[0]]  # close the shape
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                name=model.split("(")[0].strip(),
                line_color=AI_SYSTEMS[model]["color"],
                fill="toself",
                opacity=0.35,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(15,37,64,0.5)",
                radialaxis=dict(gridcolor=GRID_COL, color="#334155"),
                angularaxis=dict(gridcolor=GRID_COL, color="#7dd3fc"),
            ),
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PAPER_BG,
            title="Multi-Dimensional Water Profile",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    <div class="assumption-box">
      <b>📌 Overview Assumptions:</b><br>
      • Daily totals combine inference cooling water + electricity-related water<br>
      • Training and development are one-time costs shown separately in the Training tab<br>
      • Radar chart scales training/dev by ÷10,000 for visual comparability with daily figures<br>
      • "Electricity" includes both on-site data center cooling (WUE) and off-site power-plant water<br>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# TAB: PER PROMPT
# ────────────────────────────────────────────────────────────────
with tab_prompt:
    st.header("Water Used Per Single Prompt")
    st.markdown(
        "Every time you send a message to an AI assistant, the data center's GPUs "
        "process your request — generating heat that requires water-based cooling."
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        prompt_vals = [to_display_unit(water[m]["inference_per_prompt_gal"])[0] for m in MODEL_NAMES]
        elec_per_prompt = [to_display_unit(water[m]["elec_per_prompt_gal"])[0]  for m in MODEL_NAMES]
        _, u_label = to_display_unit(1)

        fig_prompt = go.Figure()
        fig_prompt.add_trace(go.Bar(
            name="Direct cooling (data center)",
            x=MODEL_NAMES, y=prompt_vals,
            marker_color=[AI_SYSTEMS[m]["color"] for m in MODEL_NAMES],
            text=[f"{v:.5f} {u_label}" for v in prompt_vals],
            textposition="outside",
            hovertemplate="%{y:.6f} " + u_label,
        ))
        if show_elec_water:
            fig_prompt.add_trace(go.Bar(
                name="Electricity water",
                x=MODEL_NAMES, y=elec_per_prompt,
                marker_color="#1e40af",
                hovertemplate="%{y:.6f} " + u_label,
            ))
        fig_prompt.update_layout(
            barmode="stack",
            title="Water per Single AI Prompt",
            yaxis_title=f"Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_prompt, use_container_width=True)

    with col_b:
        st.markdown("#### How does it scale with usage?")
        scale_max = st.slider("Max prompts to chart", 10, 1000, 100, key="scale_slider")
        x_range = list(range(1, scale_max + 1))

        fig_scale = go.Figure()
        for model in MODEL_NAMES:
            y_vals = [
                to_display_unit(water[model]["inference_per_prompt_gal"] * n)[0]
                for n in x_range
            ]
            fig_scale.add_trace(go.Scatter(
                x=x_range, y=y_vals,
                name=model.split("(")[0].strip(),
                line=dict(color=AI_SYSTEMS[model]["color"], width=2),
                mode="lines",
                hovertemplate="%{y:.4f} " + u_label,
            ))
        fig_scale.update_layout(
            title="Cumulative Water vs. Number of Prompts",
            xaxis_title="Number of Prompts",
            yaxis_title=f"Total Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
            xaxis=dict(gridcolor=GRID_COL),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    # Monthly / annual projections
    st.markdown("#### 📅 Your Personal Usage Projections")
    proj_cols = st.columns(len(MODEL_NAMES))
    for col, model in zip(proj_cols, MODEL_NAMES):
        with col:
            w_per_day  = water[model]["daily_total_gal"]
            w_per_month = w_per_day * 30
            w_per_year  = w_per_day * 365
            st.markdown(f"**{model.split('(')[0].strip()}**")
            st.write(f"Daily:  {fmt_water(w_per_day)}")
            st.write(f"Monthly: {fmt_water(w_per_month)}")
            st.write(f"Annual:  {fmt_water(w_per_year)}")

    st.markdown("""
    <div class="assumption-box">
      <b>📌 Per-Prompt Assumptions:</b><br>
      • ChatGPT baseline: ~500 mL per 20 prompts = 25 mL/prompt (Li et al. 2023) at US data centers<br>
      • Claude, Gemini, and Llama baselines are extrapolated from model scale and data center WUE; no official figures available<br>
      • Llama 3 is estimated lower due to smaller model size (70B vs ~1T+ parameters for GPT-4 class) and Meta's lower WUE<br>
      • Override any of these in the sidebar to test your own estimates<br>
      • Values represent <em>direct</em> on-site cooling water; check "Include electricity-related water" for full picture
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# TAB: TRAINING & DEVELOPMENT
# ────────────────────────────────────────────────────────────────
with tab_training:
    st.header("Training & Development Water")
    st.markdown(
        "Training a large AI model is a **massive, one-time event** consuming "
        "energy and water for weeks or months. Development (fine-tuning, safety "
        "testing, RLHF) adds further cost before a model is released."
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        _, u_label = to_display_unit(1)
        train_vals = [to_display_unit(water[m]["training_total_gal"])[0] for m in MODEL_NAMES]
        dev_vals   = [to_display_unit(water[m]["dev_total_gal"])[0]       for m in MODEL_NAMES]

        fig_train = go.Figure()
        fig_train.add_trace(go.Bar(
            name="Training (initial run)",
            x=MODEL_NAMES, y=train_vals,
            marker_color="#0ea5e9",
            hovertemplate="%{y:,.0f} " + u_label,
        ))
        fig_train.add_trace(go.Bar(
            name="Development (RLHF, fine-tuning, evals)",
            x=MODEL_NAMES, y=dev_vals,
            marker_color="#075985",
            hovertemplate="%{y:,.0f} " + u_label,
        ))
        fig_train.update_layout(
            barmode="group",
            title="One-Time Training & Development Water Cost",
            yaxis_title=f"Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_train, use_container_width=True)

    with col_b:
        st.markdown("#### Amortized Per-User, Per-Day Share")
        st.caption("One-time training costs divided across all users and your amortization window.")

        user_count = st.select_slider(
            "Daily active users (global)",
            options=[1_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000],
            value=100_000_000,
            format_func=lambda x: f"{x / 1_000_000:.0f}M",
            key="user_count_slider",
        )

        # Per-user daily share: total / (years × 365 days × users)
        amort_vals = [
            to_display_unit(
                water[m]["training_total_gal"] / (amortization_years * 365 * user_count)
            )[0]
            for m in MODEL_NAMES
        ]

        fig_amort = go.Figure(go.Bar(
            x=MODEL_NAMES, y=amort_vals,
            marker_color=[AI_SYSTEMS[m]["color"] for m in MODEL_NAMES],
            text=[f"{v:.8f} {u_label}" for v in amort_vals],
            textposition="outside",
            hovertemplate="%{y:.8f} " + u_label,
        ))
        fig_amort.update_layout(
            title=f"Training Water Per User / Day\n({user_count/1e6:.0f}M users · {amortization_years}yr amort.)",
            yaxis_title=f"Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
        )
        st.plotly_chart(fig_amort, use_container_width=True)

    # Context: put training water in human terms
    st.markdown("#### 🧊 What Does Training Water Actually Look Like?")
    ctx_cols = st.columns(len(MODEL_NAMES))
    for col, model in zip(ctx_cols, MODEL_NAMES):
        with col:
            tg = water[model]["training_total_gal"]
            pools  = tg / 660_000           # Olympic pool ≈ 660K gallons
            drink  = tg / (0.5 * 365)       # person drinks ~0.5 gal/day
            showers = tg / (2.0 * 8)        # EPA: ~2 gal/min × 8 min avg shower

            st.markdown(f"**{model.split('(')[0].strip()}**")
            st.markdown(f"🏊 **{pools:.2f}** Olympic pools")
            st.markdown(f"🧑 **{drink:,.0f}** person-years of drinking")
            st.markdown(f"🚿 **{showers:,.0f}** average showers")
            st.caption(f"≈ {fmt_water(tg)} total")

    st.markdown("""
    <div class="assumption-box">
      <b>📌 Training & Development Assumptions:</b><br>
      • GPT-3 training ≈ 700,000 L is the most-cited published estimate (Li et al. 2023)<br>
      • GPT-4: 5× GPT-3 multiplier applied; <em>OpenAI has not disclosed training compute or water</em><br>
      • Claude & Gemini: scaled from published model size descriptions; no official disclosure from either company<br>
      • Development water = training water × dev fraction; default fractions are 20–30% based on ML engineering literature<br>
      • Amortized per-user figure assumes simultaneous global use, which is a simplification (training is a sunk cost shared across all current and future users)<br>
      • Olympic pool ≈ 660,000 gal | average shower ≈ 16 gal (2 gal/min × 8 min) | person drinks ≈ 0.5 gal/day
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# TAB: ELECTRICITY
# ────────────────────────────────────────────────────────────────
with tab_electricity:
    st.header("Water from Electricity Use")
    st.markdown(
        "Most AI data centers rely at least partly on the US grid, where "
        "thermoelectric power plants (coal, gas, nuclear) withdraw large amounts "
        "of water for cooling. Even 'clean' nuclear power is very water-intensive."
    )

    col_a, col_b = st.columns([3, 2])
    _, u_label = to_display_unit(1)

    with col_a:
        # Breakdown: on-site WUE cooling vs off-site power-plant water
        dc_cool = [
            to_display_unit(
                AI_SYSTEMS[m]["power_per_prompt_kwh"]
                * AI_SYSTEMS[m]["data_center_wue"]
                * LITERS_TO_GALLONS
                * daily_prompts
            )[0]
            for m in MODEL_NAMES
        ]
        offsite = [
            to_display_unit(
                AI_SYSTEMS[m]["power_per_prompt_kwh"] * elec_intensity * daily_prompts
            )[0] if show_elec_water else 0.0
            for m in MODEL_NAMES
        ]

        fig_elec = go.Figure()
        fig_elec.add_trace(go.Bar(
            name="Data center cooling (on-site WUE)",
            x=MODEL_NAMES, y=dc_cool,
            marker_color="#0ea5e9",
        ))
        fig_elec.add_trace(go.Bar(
            name="Power plant water (off-site grid)",
            x=MODEL_NAMES, y=offsite,
            marker_color="#1e3a5f",
        ))
        fig_elec.update_layout(
            barmode="stack",
            title=f"Electricity-Related Water ({daily_prompts} prompts/day)",
            yaxis_title=f"Water ({u_label})",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_elec, use_container_width=True)

    with col_b:
        # WUE comparison chart
        wue_companies = ["Microsoft Azure", "Amazon AWS", "Meta", "Anthropic (avg)", "Google GCP"]
        wue_values    = [0.49, 0.18, 0.26, 0.51, 1.0]
        wue_colors    = ["#10b981", "#f97316", "#a855f7", "#f97316", "#4285f4"]
        wue_sources   = [
            "2022 Sustainability Report",
            "2022 Sustainability Report",
            "2022 Sustainability Report",
            "Weighted AWS/GCP avg",
            "2022 Environmental Report",
        ]

        # Sort by WUE for readability
        sorted_pairs = sorted(zip(wue_values, wue_companies, wue_colors), key=lambda x: x[0])
        s_vals, s_names, s_colors = zip(*sorted_pairs)

        fig_wue = go.Figure(go.Bar(
            x=s_names, y=s_vals,
            marker_color=list(s_colors),
            text=[f"{v} L/kWh" for v in s_vals],
            textposition="outside",
        ))
        fig_wue.update_layout(
            title="Data Center WUE by Company",
            yaxis_title="WUE (Liters water / kWh)",
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            yaxis=dict(gridcolor=GRID_COL),
        )
        st.plotly_chart(fig_wue, use_container_width=True)
        st.caption("Lower WUE = more water efficient. AWS leads at 0.18 L/kWh; GCP is highest at ~1.0 L/kWh.")

    # Power-per-prompt summary table
    st.markdown("#### Estimated Compute & Power per Prompt")
    power_df = pd.DataFrame({
        "Model":                MODEL_NAMES,
        "Company":              [AI_SYSTEMS[m]["company"] for m in MODEL_NAMES],
        "Power/Prompt (kWh)":   [AI_SYSTEMS[m]["power_per_prompt_kwh"] for m in MODEL_NAMES],
        "Daily Power (kWh)":    [AI_SYSTEMS[m]["power_per_prompt_kwh"] * daily_prompts for m in MODEL_NAMES],
        "Data Center WUE":      [f"{AI_SYSTEMS[m]['data_center_wue']} L/kWh" for m in MODEL_NAMES],
        "Disclosed?":           ["No — estimate" for _ in MODEL_NAMES],
    })
    st.dataframe(power_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="assumption-box">
      <b>📌 Electricity Water Assumptions:</b><br>
      • WUE figures from published 2022–2023 corporate sustainability reports<br>
      • Power per prompt: 0.002–0.003 kWh based on GPU inference benchmarks; no company has officially disclosed this<br>
      • US grid water intensity: 1.8 gal/kWh withdrawn · 0.3 gal/kWh consumed (USGS)<br>
      • Withdrawn vs. consumed: most power-plant water is returned to its source (rivers, lakes) — consumed is the
        ecologically meaningful figure but withdrawn indicates total pressure on water bodies<br>
      • Renewables (solar PV, wind) use far less water for generation; data centers with 100% renewable PPAs may have
        much lower off-site electricity water footprints
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# TAB: REAL-WORLD SCALE
# ────────────────────────────────────────────────────────────────
with tab_realworld:
    st.header("Putting It in Perspective")
    st.markdown("Numbers only become meaningful when you can feel their scale. Let's translate.")

    # ── Personal comparison ──────────────────────────────────────
    st.markdown("### 🧍 Your Personal Footprint")

    sel_model = st.selectbox("Select AI system", MODEL_NAMES, key="rw_model")
    scope = st.radio(
        "Compare water for:",
        ["Per prompt", "Your daily usage", "Model training (lifetime total)"],
        horizontal=True,
    )

    if scope == "Per prompt":
        gal_val = water[sel_model]["inference_per_prompt_gal"]
        scope_label = "per prompt"
    elif scope == "Your daily usage":
        gal_val = water[sel_model]["daily_total_gal"]
        scope_label = f"per day ({daily_prompts} prompts)"
    else:
        gal_val = water[sel_model]["training_total_gal"]
        scope_label = "total model training"

    st.metric(
        f"Water: {sel_model.split('(')[0].strip()} — {scope_label}",
        fmt_water(gal_val),
    )

    # Comparison cards
    card_cols = st.columns(3)
    card_idx = 0
    for (icon, label), gal_each in WATER_BENCHMARKS.items():
        count = gal_val / gal_each
        # Only show if count is within a humanly-useful display range
        if 0.001 <= count <= 10_000_000:
            with card_cols[card_idx % 3]:
                if count < 0.01:       count_str = f"{count:.4f}"
                elif count < 1:        count_str = f"{count:.2f}"
                elif count < 100:      count_str = f"{count:.1f}"
                elif count < 100_000:  count_str = f"{count:,.0f}"
                else:                  count_str = f"{count/1_000:.1f}K"

                st.markdown(f"""
                <div class="comparison-card">
                  <div class="icon">{icon}</div>
                  <div class="count">{count_str}</div>
                  <div class="label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            card_idx += 1

    st.markdown("---")

    # ── Global scale ─────────────────────────────────────────────
    st.markdown("### 🌐 At Global Scale — ChatGPT")
    st.caption(
        "ChatGPT reportedly had ~100M+ daily active users as of early 2023. "
        "What does that mean for aggregate water use?"
    )

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        global_users_m = st.slider("Global daily active users (millions)", 10, 500, 100)
    with g_col2:
        avg_prompts_gbl = st.slider("Average prompts per user per day", 1, 50, 10)

    global_daily_gal = (
        water["ChatGPT (GPT-4)"]["inference_per_prompt_gal"]
        * avg_prompts_gbl
        * global_users_m * 1_000_000
    )

    g_col_a, g_col_b, g_col_c, g_col_d = st.columns(4)
    with g_col_a:
        st.metric("Daily inference water", fmt_water(global_daily_gal))
    with g_col_b:
        st.metric("Olympic pools", f"{global_daily_gal / 660_000:,.1f}")
    with g_col_c:
        st.metric("Person-days of drinking water", f"{global_daily_gal / 0.5:,.0f}")
    with g_col_d:
        st.metric("Average showers", f"{global_daily_gal / 16:,.0f}")

    # Global timeline chart
    days = list(range(1, 366))
    global_cumulative = [
        to_display_unit(global_daily_gal * d)[0] for d in days
    ]
    _, u_label = to_display_unit(1)

    fig_global = go.Figure(go.Scatter(
        x=days,
        y=global_cumulative,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#38bdf8", width=2),
        fillcolor="rgba(56, 189, 248, 0.1)",
        hovertemplate="Day %{x}: %{y:,.0f} " + u_label,
    ))
    fig_global.update_layout(
        title=f"Cumulative Global ChatGPT Inference Water Over One Year\n({global_users_m}M users · {avg_prompts_gbl} prompts/day)",
        xaxis_title="Day of Year",
        yaxis_title=f"Cumulative Water ({u_label})",
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        yaxis=dict(gridcolor=GRID_COL),
        xaxis=dict(gridcolor=GRID_COL),
    )
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("""
    <div class="assumption-box">
      <b>📌 Real-World Comparison Benchmarks:</b><br>
      • 8 oz drinking glass = 0.0625 gal | 16.9 oz bottle = 0.132 gal<br>
      • Cup of coffee (brewed) ≈ 4 oz water = 0.031 gal<br>
      • Toilet flush = 1.6 gal (EPA WaterSense standard)<br>
      • Shower = 2 gal/min × 8 min avg = ~16 gal (EPA)<br>
      • Bathtub = 36 gal (US average fill)<br>
      • Olympic pool = ~2.5M liters = ~660,000 gal<br>
      • Person's daily drinking water = ~0.5 gal/day (US average)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 8: FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-text">
  <b>💧 AI Water Usage Explorer</b> &nbsp;|&nbsp; Built for public education &nbsp;|&nbsp; Data as of 2023–2024<br>
  Primary source: <em>Li, P. et al. (2023) "Making AI Less Thirsty: Uncovering and Addressing the Secret Water Footprint of AI Models"</em><br>
  Also drawing on: Microsoft, Google, Amazon & Meta Sustainability Reports (2022) · USGS National Water-Use Science Project<br><br>
  <em>
    Transparency: The majority of AI companies do not publicly disclose water usage data at the model or prompt level.
    All figures on this dashboard are derived estimates intended as order-of-magnitude approximations.
    They should not be cited as official measurements. All assumptions are documented in-app.
  </em>
</div>
""", unsafe_allow_html=True)
