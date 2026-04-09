"""
Compressor EDA — KES22_8p5 Synthetic Test Data
Goal: Understand sensor behaviour across test phases to support
      test-time reduction and risk-score modelling.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FILE   = r"C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\data\KES22_8p5_synthetic_test_data.xlsx"
OUTDIR = r"C:\Users\Admin\OneDrive - C4i4 Lab, Samarth Udyog Technology Forum,\Attachments\Office work\DigitalTwin2\outputs"
os.makedirs(OUTDIR, exist_ok=True)

PHASE_COLORS = {
    "Phase1_Warmup":        "#F4A460",
    "Phase2_Stabilization": "#4FC3F7",
    "Phase3_StableRated":   "#66BB6A",
    "Phase4_UnloadCycle":   "#EF5350",
}
PHASE_ORDER = ["Phase1_Warmup","Phase2_Stabilization",
               "Phase3_StableRated","Phase4_UnloadCycle"]

TEMP_COLS = ["airend_discharge_temp_c","oil_cooler_inlet_temp_c",
             "oil_cooler_outlet_temp_c","aftercooler_inlet_temp_c",
             "aftercooler_outlet_temp_c","air_inlet_temp_c"]

PRESSURE_COLS = ["delivery_pressure_kg_cm2g","aos_tank_inlet_pressure_kg_cm2g"]

POWER_COLS = ["motor_output_power_kw","package_input_power_kw",
              "power_factor","current_package_input_a","input_voltage_v"]

PERF_COLS  = ["fad_cfm","spc_kw_per_m3_min",
              "tolerance_flow_pct","tolerance_spc_pct"]

ALL_SENSOR = TEMP_COLS + PRESSURE_COLS + POWER_COLS + PERF_COLS

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

# ─── LOAD ─────────────────────────────────────────────────────────────────────
df = pd.read_excel(FILE)
df["elapsed_time_min"] = df["elapsed_time_min"].astype(float)
df["phase"] = pd.Categorical(df["phase"], categories=PHASE_ORDER, ordered=True)

# phase boundaries for shading
phase_bounds = (df.groupby("phase", observed=True)["elapsed_time_min"]
                  .agg(["min","max"]).loc[PHASE_ORDER])

def shade_phases(ax):
    for phase in PHASE_ORDER:
        if phase in phase_bounds.index:
            ax.axvspan(phase_bounds.loc[phase,"min"],
                       phase_bounds.loc[phase,"max"],
                       alpha=0.08, color=PHASE_COLORS[phase])

def phase_legend():
    return [mpatches.Patch(color=PHASE_COLORS[p], label=p.replace("_"," "), alpha=0.6)
            for p in PHASE_ORDER]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold", y=1.02)

# Phase distribution — bar
phase_cnt = df["phase"].value_counts().reindex(PHASE_ORDER)
colors     = [PHASE_COLORS[p] for p in PHASE_ORDER]
axes[0].bar([p.replace("_","\n") for p in PHASE_ORDER], phase_cnt.values, color=colors, edgecolor="white")
axes[0].set_title("Sample Count per Phase")
axes[0].set_ylabel("# Readings")

# Phase duration pie
durations = phase_bounds["max"] - phase_bounds["min"]
axes[1].pie(durations, labels=[p.replace("_","\n") for p in PHASE_ORDER],
            colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Phase Share of Total Test Time (~180 min)")

plt.tight_layout()
fig.savefig(f"{OUTDIR}/01_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALL SENSORS OVER TIME  (4 sub-panels)
# ═══════════════════════════════════════════════════════════════════════════════
sensor_groups = [
    ("Temperature Sensors (°C)", TEMP_COLS),
    ("Pressure Sensors (kg/cm²g)", PRESSURE_COLS),
    ("Power & Electrical", POWER_COLS),
    ("Performance KPIs", PERF_COLS),
]

for panel_title, cols in sensor_groups:
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3*n), sharex=True)
    fig.suptitle(f"Sensor Readings Over Time — {panel_title}", fontsize=13, fontweight="bold")
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for phase in PHASE_ORDER:
            sub = df[df["phase"]==phase]
            ax.plot(sub["elapsed_time_min"], sub[col],
                    color=PHASE_COLORS[phase], linewidth=1.2, alpha=0.9)
        shade_phases(ax)
        ax.set_ylabel(col.replace("_"," ").replace(" c"," °C"), fontsize=9)
        ax.set_title(col.replace("_"," ").title(), fontsize=10)
    axes[-1].set_xlabel("Elapsed Time (min)")
    fig.legend(handles=phase_legend(), loc="upper right",
               bbox_to_anchor=(1.13, 0.98), framealpha=0.9)
    safe = panel_title.split("—")[-1].strip().split("(")[0].strip().replace(" ","_").lower()
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/02_timeseries_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROLLING MEAN  (stability view, window = 20 readings)
# ═══════════════════════════════════════════════════════════════════════════════
KEY_SENSORS = ["airend_discharge_temp_c","delivery_pressure_kg_cm2g",
               "fad_cfm","package_input_power_kw","spc_kw_per_m3_min","power_factor"]

fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
fig.suptitle("Rolling Mean (Window=20) — Key Sensors for Stability Detection",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for ax, col in zip(axes, KEY_SENSORS):
    ax.plot(df["elapsed_time_min"], df[col],
            alpha=0.25, color="steelblue", linewidth=0.8, label="Raw")
    roll = df[col].rolling(20, center=True).mean()
    ax.plot(df["elapsed_time_min"], roll,
            color="crimson", linewidth=2.0, label="Rolling Mean")
    shade_phases(ax)
    ax.set_title(col.replace("_"," ").title(), fontsize=10)
    ax.set_ylabel("")
    ax.legend(fontsize=8)

for ax in axes: ax.set_xlabel("Elapsed Time (min)")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/03_rolling_mean_stability.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PHASE-WISE BOXPLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def make_boxplots(cols, title, fname):
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        data  = [df[df["phase"]==p][col].dropna() for p in PHASE_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, phase in zip(bp["boxes"], PHASE_ORDER):
            patch.set_facecolor(PHASE_COLORS[phase])
            patch.set_alpha(0.7)
        ax.set_xticklabels([p.split("_")[0]+"\n"+p.split("_",1)[1]
                            for p in PHASE_ORDER], fontsize=8)
        ax.set_title(col.replace("_"," ").title(), fontsize=9)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()

make_boxplots(TEMP_COLS,     "Phase-wise Boxplots — Temperatures",  "04a_boxplots_temp")
make_boxplots(PRESSURE_COLS+POWER_COLS, "Phase-wise Boxplots — Pressures & Power", "04b_boxplots_pressure_power")
make_boxplots(PERF_COLS,     "Phase-wise Boxplots — Performance KPIs","04c_boxplots_perf")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
corr = df[ALL_SENSOR].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 7}, cbar_kws={"shrink":0.8})
ax.set_title("Correlation Matrix — All Sensors", fontsize=14, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTION HISTOGRAMS  (Phase3 stable rated — the golden zone)
# ═══════════════════════════════════════════════════════════════════════════════
df3 = df[df["phase"]=="Phase3_StableRated"]
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
fig.suptitle("Sensor Distributions During Phase 3 (Stable Rated) — The Reference Zone",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
for ax, col in zip(axes, ALL_SENSOR):
    ax.hist(df3[col].dropna(), bins=25, color="#66BB6A", edgecolor="white", alpha=0.8)
    ax.axvline(df3[col].mean(), color="crimson", lw=1.5, linestyle="--", label="Mean")
    ax.set_title(col.replace("_"," ").title(), fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes[len(ALL_SENSOR):]: ax.set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/06_distributions_phase3.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 7. OUTLIER DETECTION  (Z-score across all phases)
# ═══════════════════════════════════════════════════════════════════════════════
zscore_df = df[ALL_SENSOR].apply(stats.zscore, nan_policy="omit").abs()
outlier_counts = (zscore_df > 3).sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Outlier Analysis (|Z-score| > 3)", fontsize=13, fontweight="bold")

outlier_counts.plot(kind="bar", ax=axes[0], color="tomato", edgecolor="white")
axes[0].set_title("Outlier Count per Sensor")
axes[0].set_ylabel("# Outlier Readings")
axes[0].set_xticklabels([c.replace("_","\n") for c in outlier_counts.index],
                         rotation=45, ha="right", fontsize=8)

# Outlier heatmap over time
fig2, ax2 = plt.subplots(figsize=(16, 5))
out_map = (df[ALL_SENSOR].apply(stats.zscore, nan_policy="omit").abs() > 3).T.astype(int)
sns.heatmap(out_map, ax=ax2, cmap="Reds", cbar_kws={"label":"Outlier"},
            yticklabels=[c.replace("_"," ") for c in ALL_SENSOR])
ax2.set_title("Outlier Occurrence Map Over Time (Red = |Z|>3)", fontsize=12)
ax2.set_xlabel("Reading Index")
plt.tight_layout()
fig2.savefig(f"{OUTDIR}/07b_outlier_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

axes[1].bar(outlier_counts.index, (zscore_df > 3).mean()*100,
            color="orange", edgecolor="white")
axes[1].set_title("Outlier % per Sensor")
axes[1].set_ylabel("% of Readings")
axes[1].set_xticklabels([c.replace("_","\n") for c in outlier_counts.index],
                         rotation=45, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/07a_outlier_counts.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PHASE TRANSITION DETAIL  (Phase2→Phase3 — when does steady state hit?)
# ═══════════════════════════════════════════════════════════════════════════════
p23 = df[df["phase"].isin(["Phase2_Stabilization","Phase3_StableRated"])].copy()
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
fig.suptitle("Phase 2 → Phase 3 Transition — Stability Onset Analysis",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
transition_sensors = ["airend_discharge_temp_c","delivery_pressure_kg_cm2g",
                      "fad_cfm","package_input_power_kw",
                      "spc_kw_per_m3_min","power_factor"]

for ax, col in zip(axes, transition_sensors):
    for ph in ["Phase2_Stabilization","Phase3_StableRated"]:
        sub = p23[p23["phase"]==ph]
        ax.plot(sub["elapsed_time_min"], sub[col],
                color=PHASE_COLORS[ph], linewidth=1.5,
                label=ph.replace("_"," "))
    roll = p23.set_index("elapsed_time_min")[col].rolling(15, center=True).mean()
    ax.plot(roll.index, roll.values, "k--", linewidth=1.5, alpha=0.5, label="Rolling Mean")
    ax.axvline(phase_bounds.loc["Phase3_StableRated","min"],
               color="green", linestyle=":", linewidth=2, label="Phase3 Start")
    ax.set_title(col.replace("_"," ").title(), fontsize=10)
    ax.legend(fontsize=7)

for ax in axes: ax.set_xlabel("Elapsed Time (min)")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/08_phase_transition_stability.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. POWER & EFFICIENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Power & Efficiency Deep-Dive", fontsize=13, fontweight="bold")

# Motor vs Package power scatter
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[0,0].scatter(sub["motor_output_power_kw"], sub["package_input_power_kw"],
                      c=PHASE_COLORS[phase], alpha=0.5, s=20, label=phase.replace("_"," "))
axes[0,0].set_xlabel("Motor Output Power (kW)")
axes[0,0].set_ylabel("Package Input Power (kW)")
axes[0,0].set_title("Motor vs Package Power")
axes[0,0].legend(fontsize=7)

# SPC over time by phase
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[0,1].plot(sub["elapsed_time_min"], sub["spc_kw_per_m3_min"],
                   color=PHASE_COLORS[phase], alpha=0.8, linewidth=1.2,
                   label=phase.replace("_"," "))
axes[0,1].set_xlabel("Elapsed Time (min)")
axes[0,1].set_ylabel("SPC (kW/m³/min)")
axes[0,1].set_title("Specific Power Consumption Over Time")
axes[0,1].legend(fontsize=7)

# Power factor over time
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[1,0].plot(sub["elapsed_time_min"], sub["power_factor"],
                   color=PHASE_COLORS[phase], alpha=0.8, linewidth=1.2,
                   label=phase.replace("_"," "))
axes[1,0].axhline(0.85, color="red", linestyle="--", label="Min Acceptable PF=0.85")
axes[1,0].set_xlabel("Elapsed Time (min)")
axes[1,0].set_ylabel("Power Factor")
axes[1,0].set_title("Power Factor Over Time")
axes[1,0].legend(fontsize=7)

# FAD vs Package power
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[1,1].scatter(sub["fad_cfm"], sub["package_input_power_kw"],
                      c=PHASE_COLORS[phase], alpha=0.5, s=20, label=phase.replace("_"," "))
axes[1,1].set_xlabel("FAD (CFM)")
axes[1,1].set_ylabel("Package Input Power (kW)")
axes[1,1].set_title("FAD vs Power (Efficiency Envelope)")
axes[1,1].legend(fontsize=7)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/09_power_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 10. THERMAL ANALYSIS — cooler effectiveness
# ═══════════════════════════════════════════════════════════════════════════════
df["oil_cooler_delta"]   = df["oil_cooler_inlet_temp_c"]   - df["oil_cooler_outlet_temp_c"]
df["aftercooler_delta"]  = df["aftercooler_inlet_temp_c"]  - df["aftercooler_outlet_temp_c"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Thermal Analysis — Cooler Effectiveness & Temperature Profile",
             fontsize=13, fontweight="bold")

# Cooler deltas over time
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[0,0].plot(sub["elapsed_time_min"], sub["oil_cooler_delta"],
                   color=PHASE_COLORS[phase], linewidth=1.2, alpha=0.9)
    axes[0,1].plot(sub["elapsed_time_min"], sub["aftercooler_delta"],
                   color=PHASE_COLORS[phase], linewidth=1.2, alpha=0.9,
                   label=phase.replace("_"," "))
axes[0,0].set_title("Oil Cooler ΔT (Inlet−Outlet) Over Time")
axes[0,0].set_ylabel("ΔT (°C)")
axes[0,1].set_title("Aftercooler ΔT (Inlet−Outlet) Over Time")
axes[0,1].legend(fontsize=7)

# Temperature profile stacked
temp_means = df.groupby("phase", observed=True)[TEMP_COLS].mean().loc[PHASE_ORDER]
temp_means.T.plot(kind="bar", ax=axes[1,0],
                  color=[PHASE_COLORS[p] for p in PHASE_ORDER],
                  edgecolor="white")
axes[1,0].set_title("Mean Temp per Sensor by Phase")
axes[1,0].set_ylabel("Temperature (°C)")
axes[1,0].set_xticklabels([c.replace("_temp_c","").replace("_"," ")
                            for c in TEMP_COLS], rotation=35, ha="right", fontsize=8)
axes[1,0].legend(fontsize=7)

# Discharge temp vs delivery pressure scatter
for phase in PHASE_ORDER:
    sub = df[df["phase"]==phase]
    axes[1,1].scatter(sub["delivery_pressure_kg_cm2g"],
                      sub["airend_discharge_temp_c"],
                      c=PHASE_COLORS[phase], alpha=0.5, s=20,
                      label=phase.replace("_"," "))
axes[1,1].set_xlabel("Delivery Pressure (kg/cm²g)")
axes[1,1].set_ylabel("Airend Discharge Temp (°C)")
axes[1,1].set_title("Pressure vs Discharge Temp")
axes[1,1].legend(fontsize=7)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/10_thermal_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PHASE-WISE SUMMARY STATISTICS TABLE (saved as CSV)
# ═══════════════════════════════════════════════════════════════════════════════
summary = df.groupby("phase", observed=True)[ALL_SENSOR].agg(["mean","std","min","max"])
summary.to_csv(f"{OUTDIR}/11_phase_summary_stats.csv")

# Visualise as heatmap of means
means = df.groupby("phase", observed=True)[ALL_SENSOR].mean().loc[PHASE_ORDER]
fig, ax = plt.subplots(figsize=(18, 5))
normed = (means - means.min()) / (means.max() - means.min() + 1e-9)
sns.heatmap(normed.T, annot=means.T.round(2), fmt=".2f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label":"Normalised Mean"},
            annot_kws={"size":7})
ax.set_title("Phase-wise Mean Values — All Sensors (Colour = Normalised)",
             fontsize=13, fontweight="bold")
ax.set_yticklabels([c.replace("_"," ") for c in ALL_SENSOR],
                   rotation=0, fontsize=8)
ax.set_xticklabels([p.replace("_","\n") for p in PHASE_ORDER], rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/11_phase_mean_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 12. COEFFICIENT OF VARIATION — which sensors stabilise first?
# ═══════════════════════════════════════════════════════════════════════════════
cv = df.groupby("phase", observed=True)[ALL_SENSOR].apply(
        lambda x: (x.std()/x.mean().abs())*100).loc[PHASE_ORDER]

fig, ax = plt.subplots(figsize=(16, 6))
cv.T.plot(kind="bar", ax=ax,
          color=[PHASE_COLORS[p] for p in PHASE_ORDER],
          edgecolor="white", width=0.7)
ax.set_title("Coefficient of Variation (%) per Phase — Lower = More Stable",
             fontsize=13, fontweight="bold")
ax.set_ylabel("CV (%)")
ax.set_xticklabels([c.replace("_"," ") for c in ALL_SENSOR],
                   rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/12_coefficient_of_variation.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 13. PAIRPLOT — Phase3 key sensors
# ═══════════════════════════════════════════════════════════════════════════════
pair_cols = ["airend_discharge_temp_c","delivery_pressure_kg_cm2g",
             "fad_cfm","package_input_power_kw","power_factor"]
pp_df = df[pair_cols + ["phase"]].copy()
pp_df["Color"] = pp_df["phase"].map(PHASE_COLORS)
g = sns.pairplot(pp_df, vars=pair_cols, hue="phase",
                 palette=PHASE_COLORS, plot_kws={"alpha":0.4,"s":15},
                 diag_kind="kde", corner=True)
g.fig.suptitle("Pairplot — Key Sensors Coloured by Phase", y=1.02,
               fontsize=13, fontweight="bold")
g.fig.savefig(f"{OUTDIR}/13_pairplot_key_sensors.png", dpi=130, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 14. ROLLING STD  — variance collapse into stable phase
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
fig.suptitle("Rolling Std (Window=20) — Variance Collapse Marks Stability",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
for ax, col in zip(axes, KEY_SENSORS):
    ax.plot(df["elapsed_time_min"],
            df[col].rolling(20, center=True).std(),
            color="purple", linewidth=1.5)
    shade_phases(ax)
    ax.set_title(col.replace("_"," ").title(), fontsize=10)
    ax.set_ylabel("Rolling Std")
for ax in axes: ax.set_xlabel("Elapsed Time (min)")
fig.legend(handles=phase_legend(), loc="upper right",
           bbox_to_anchor=(1.13,0.98), framealpha=0.9)
plt.tight_layout()
fig.savefig(f"{OUTDIR}/14_rolling_std_variance.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 15. CUMULATIVE MEAN convergence — earliest stable window candidate
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
fig.suptitle("Cumulative Mean — Convergence Shows When Test Can Potentially Stop",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for ax, col in zip(axes, KEY_SENSORS):
    cum_mean  = df[col].expanding().mean()
    phase3_mean = df[df["phase"]=="Phase3_StableRated"][col].mean()
    ax.plot(df["elapsed_time_min"], cum_mean,
            color="steelblue", linewidth=1.5, label="Cumulative Mean")
    ax.axhline(phase3_mean, color="green", linestyle="--",
               linewidth=1.5, label="Phase3 Final Mean")
    shade_phases(ax)
    ax.set_title(col.replace("_"," ").title(), fontsize=10)
    ax.legend(fontsize=7)

for ax in axes: ax.set_xlabel("Elapsed Time (min)")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/15_cumulative_mean_convergence.png", dpi=150, bbox_inches="tight")
plt.close()

print("✅  All 15 EDA plots saved to", OUTDIR)
print("📊  Files generated:")
import glob
for f in sorted(glob.glob(f"{OUTDIR}/*.png")):
    print("   ", os.path.basename(f))
print(f"   11_phase_summary_stats.csv")