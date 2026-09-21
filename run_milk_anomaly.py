"""
Milk Anomaly Detection System
==============================
Detects anomalies in robotic milking parlor data using rolling-window
Z-scores, consecutive-day streak scoring, composite health metrics,
and optional rumination data integration.

Outputs:
  - daily_milk_anomalies.csv   – per-cow daily features + anomaly flags
  - problem_cows_summary.csv   – cow-level anomaly summary
  - daily_alert_report.md      – actionable Markdown alert report
  - plots/                     – per-cow trend visualisation PNGs
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / headless use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────────
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_ROLLING_WINDOW = 14          # days
DEFAULT_MIN_WINDOW_DAYS = 5          # minimum days before rolling stats kick in
DEFAULT_COMPOSITE_WEIGHTS = {
    "milk": 0.35,
    "fat": 0.20,
    "protein": 0.10,
    "fat_protein_ratio": 0.15,
    "blood": 0.20,
}

# Absolute thresholds for blood and F:P ratio (not Z-score based)
BLOOD_WARNING_THRESHOLD = 1.0     # blood indicator > 1 → warning
BLOOD_SERIOUS_THRESHOLD = 2.0     # blood indicator > 2 → serious
FPR_WARNING_THRESHOLD = 1.4       # fat:protein ratio > 1.4 → ketosis warning
FPR_SERIOUS_THRESHOLD = 1.5       # fat:protein ratio > 1.5 → serious ketosis risk


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════

def load_and_clean(input_csv: str) -> pd.DataFrame:
    """Load the robot CSV, fix Italian number formatting, and filter to
    successful milkings with plausible inter-milking intervals."""

    log.info("Loading %s …", input_csv)
    df = pd.read_csv(input_csv, sep=";", encoding="latin1")

    # ── Input validation ────────────────────────────────────────────────
    required_cols = {"Vacca", "Data", "Ora", "Latte. kg", "Grasso. %",
                     "Sangue", "Tipo Mung", "Prot %", "Lattosio",
                     "Rapporto G/P", "Data parto", "GIM"}
    missing = required_cols - set(df.columns)
    if missing:
        log.warning("Missing columns (will be skipped): %s", missing)

    # ── Drop phantom column (extra semicolon between Latte and Grasso) ──
    phantom_cols = [c for c in df.columns if "Unnamed" in str(c)]
    if phantom_cols:
        log.info("Dropping phantom columns: %s", phantom_cols)
        df = df.drop(columns=phantom_cols)

    # ── Helper to clean Italian-formatted numeric columns ───────────────
    def _clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace("\t", "", regex=False)
            .str.replace("-", "", regex=False),
            errors="coerce",
        )

    # ── Clean numeric columns ───────────────────────────────────────────
    df["Latte. kg"] = _clean_numeric(df["Latte. kg"])

    for col in ["Grasso. %", "Prot %", "Lattosio", "Rapporto G/P", "Sangue"]:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    if "Sangue" in df.columns:
        df["Sangue"] = df["Sangue"].fillna(0)

    if "GIM" in df.columns:
        df["GIM"] = pd.to_numeric(df["GIM"], errors="coerce")

    # ── Rename core columns ─────────────────────────────────────────────
    rename_map = {
        "Vacca": "cow_id",
        "Data": "date",
        "Latte. kg": "milk_yield",
        "Grasso. %": "fat_pct",
        "Prot %": "protein_pct",
        "Lattosio": "lactose_pct",
        "Rapporto G/P": "fat_protein_ratio",
        "Sangue": "blood",
        "GIM": "dim",           # Days In Milk
        "Data parto": "calving_date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Parse dates ─────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["Ora"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    if "calving_date" in df.columns:
        df["calving_date"] = pd.to_datetime(df["calving_date"], dayfirst=True, errors="coerce")

    # ── Filter to successful milkings ───────────────────────────────────
    before = len(df)
    df = df[df["Tipo Mung"] == "Ok"].copy()
    df = df[df["milk_yield"] > 0]
    log.info("Kept %d / %d rows after filtering unsuccessful milkings.", len(df), before)

    # ── Compute inter-milking interval and milk rate ────────────────────
    df = df.sort_values(["cow_id", "datetime"])
    df["hours_since_last"] = (
        df.groupby("cow_id")["datetime"]
        .diff()
        .dt.total_seconds() / 3600
    )
    df = df[(df["hours_since_last"] > 0) & (df["hours_since_last"] < 24)]
    df["milk_rate"] = df["milk_yield"] / df["hours_since_last"]

    # ── Compute DIM from calving date if the column didn't exist ────────
    if "dim" not in df.columns and "calving_date" in df.columns:
        df["dim"] = (df["date"] - df["calving_date"]).dt.days

    n_cows = df["cow_id"].nunique()
    date_range = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
    log.info("Cleaned data: %d milking events, %d cows, date range %s", len(df), n_cows, date_range)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD RUMINATION DATA (optional)
# ═══════════════════════════════════════════════════════════════════════════

def load_rumination(rumination_path: str | None) -> pd.DataFrame | None:
    """Load and aggregate rumination Excel data to daily level per cow."""
    if rumination_path is None or not os.path.exists(rumination_path):
        log.info("No rumination data provided — skipping.")
        return None

    log.info("Loading rumination data from %s …", rumination_path)
    rum = pd.read_excel(rumination_path)

    rum = rum.rename(columns={
        "Numero Vacca": "cow_id",
        "Data": "date",
        "Ruminazione Giornaliera": "rumination_daily",
        "Ruminazione Media Settimanale": "rumination_weekly_avg",
        "Variazione Ruminazione Calcolata": "rumination_variation",
    })

    rum["date"] = pd.to_datetime(rum["date"], errors="coerce")

    # Aggregate to one row per (cow, date) — use first observation since
    # daily/weekly columns repeat within the same day.
    daily_rum = (
        rum.groupby(["cow_id", "date"])
        .agg(
            rumination_daily=("rumination_daily", "first"),
            rumination_weekly_avg=("rumination_weekly_avg", "first"),
            rumination_variation=("rumination_variation", "first"),
        )
        .reset_index()
    )

    log.info("Rumination data: %d cow-days loaded.", len(daily_rum))
    return daily_rum


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPUTE DAILY FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_daily_features(
    df: pd.DataFrame,
    rumination: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate milking-level data to daily per-cow features and optionally
    merge rumination data."""

    agg_dict = {
        "milk_rate": ("milk_rate", "median"),
        "fat_pct": ("fat_pct", "median"),
        "protein_pct": ("protein_pct", "median"),
        "lactose_pct": ("lactose_pct", "median"),
        "fat_protein_ratio": ("fat_protein_ratio", "median"),
        "blood": ("blood", "max"),    # max, not median — any bleeding event matters
        "milkings_today": ("milk_rate", "count"),
        "total_milk_yield": ("milk_yield", "sum"),
    }

    # Include DIM if available
    if "dim" in df.columns:
        agg_dict["dim"] = ("dim", "first")

    # Only keep columns that exist
    available_agg = {}
    for key, (col, func) in agg_dict.items():
        if col in df.columns:
            available_agg[key] = (col, func)

    daily = (
        df.groupby(["cow_id", "date"])
        .agg(**available_agg)
        .reset_index()
    )

    # ── Merge rumination data if available ──────────────────────────────
    if rumination is not None:
        daily = daily.merge(rumination, on=["cow_id", "date"], how="left")
        log.info("Merged rumination data into daily features.")

    log.info("Daily features computed: %d cow-days.", len(daily))
    return daily


# ═══════════════════════════════════════════════════════════════════════════
# 4. DETECT ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(
    daily: pd.DataFrame,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS,
    composite_weights: dict | None = None,
) -> pd.DataFrame:
    """Add rolling Z-scores, anomaly flags, consecutive-day streaks,
    and a composite health score to the daily DataFrame."""

    weights = composite_weights or DEFAULT_COMPOSITE_WEIGHTS

    # ── Rolling Z-scores per cow (only for metrics where relative
    #    deviation matters — NOT blood or F:P ratio) ─────────────────────
    z_cols = {
        "milk_rate": "milk_z",
        "fat_pct": "fat_z",
        "protein_pct": "protein_z",
        # blood and fat_protein_ratio use absolute thresholds, not Z-scores
    }

    daily = daily.sort_values(["cow_id", "date"])

    for src_col, z_col in z_cols.items():
        if src_col not in daily.columns:
            continue

        rolling_mean = (
            daily.groupby("cow_id")[src_col]
            .transform(lambda x: x.rolling(rolling_window, min_periods=min_window_days).mean())
        )
        rolling_std = (
            daily.groupby("cow_id")[src_col]
            .transform(lambda x: x.rolling(rolling_window, min_periods=min_window_days).std())
        )
        # Avoid division by zero — use NaN where std is 0
        daily[z_col] = (daily[src_col] - rolling_mean) / rolling_std.replace(0, np.nan)

    # ── Rumination Z-score if available ─────────────────────────────────
    if "rumination_daily" in daily.columns:
        rolling_mean = (
            daily.groupby("cow_id")["rumination_daily"]
            .transform(lambda x: x.rolling(rolling_window, min_periods=min_window_days).mean())
        )
        rolling_std = (
            daily.groupby("cow_id")["rumination_daily"]
            .transform(lambda x: x.rolling(rolling_window, min_periods=min_window_days).std())
        )
        daily["rumination_z"] = (daily["rumination_daily"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # ── Binary anomaly flags (Z-score based) ──────────────────────────
    z_flag_map = {
        "milk_z": ("milk_anomaly", "low_milk", "high_milk"),
        "fat_z": ("fat_anomaly", "low_fat", "high_fat"),
        "protein_z": ("protein_anomaly", "low_protein", "high_protein"),
    }

    for z_col, (anomaly, low, high) in z_flag_map.items():
        if z_col not in daily.columns:
            continue
        daily[anomaly] = daily[z_col].abs() > z_threshold
        if low:
            daily[low] = daily[z_col] < -z_threshold
        if high:
            daily[high] = daily[z_col] > z_threshold

    if "rumination_z" in daily.columns:
        daily["rumination_anomaly"] = daily["rumination_z"] < -z_threshold  # low rumination is bad

    # ── Blood: absolute threshold (any bleeding matters) ────────────────
    if "blood" in daily.columns:
        daily["blood_warning"]  = daily["blood"] > BLOOD_WARNING_THRESHOLD
        daily["blood_serious"]  = daily["blood"] > BLOOD_SERIOUS_THRESHOLD
        daily["blood_anomaly"]  = daily["blood_warning"]  # any level counts as anomaly

    # ── F:P ratio: absolute threshold (ketosis risk) ───────────────────
    if "fat_protein_ratio" in daily.columns:
        daily["fpr_warning"]  = daily["fat_protein_ratio"] > FPR_WARNING_THRESHOLD
        daily["fpr_serious"]  = daily["fat_protein_ratio"] > FPR_SERIOUS_THRESHOLD
        daily["fpr_anomaly"]  = daily["fpr_warning"]  # any level counts as anomaly
        daily["ketosis_risk"] = daily["fpr_serious"]   # backward-compatible alias

    # ── Composite health score ──────────────────────────────────────────
    score_parts = []
    weight_sum = 0.0

    # Z-score based components
    z_weight_map = {
        "milk": "milk_z",
        "fat": "fat_z",
        "protein": "protein_z",
    }
    for key, z_col in z_weight_map.items():
        if z_col in daily.columns and key in weights:
            score_parts.append(daily[z_col].abs().fillna(0) * weights[key])
            weight_sum += weights[key]

    # Blood: use raw value normalized (0→0, ≥2→1) so it scales with Z-scores
    if "blood" in daily.columns and "blood" in weights:
        blood_norm = (daily["blood"] / BLOOD_SERIOUS_THRESHOLD).clip(0, 2).fillna(0)
        score_parts.append(blood_norm * weights["blood"])
        weight_sum += weights["blood"]

    # F:P ratio: distance above warning threshold, normalized
    if "fat_protein_ratio" in daily.columns and "fat_protein_ratio" in weights:
        fpr_excess = ((daily["fat_protein_ratio"] - FPR_WARNING_THRESHOLD) / 0.2).clip(0, 2).fillna(0)
        score_parts.append(fpr_excess * weights["fat_protein_ratio"])
        weight_sum += weights["fat_protein_ratio"]

    if "rumination_z" in daily.columns:
        # Rumination drop is concerning → use negative z (so abs of negative)
        rum_w = 0.15  # extra weight for rumination
        score_parts.append(daily["rumination_z"].clip(upper=0).abs().fillna(0) * rum_w)
        weight_sum += rum_w

    if score_parts:
        daily["health_score"] = sum(score_parts) / weight_sum
    else:
        daily["health_score"] = np.nan

    # ── Consecutive anomaly day streaks ─────────────────────────────────
    for flag_col in ["milk_anomaly", "fat_anomaly", "blood_anomaly"]:
        if flag_col not in daily.columns:
            continue
        streak_col = flag_col.replace("_anomaly", "_streak")
        streaks = []
        for _, cow_df in daily.groupby("cow_id"):
            s = cow_df[flag_col].astype(int)
            # cumsum trick: reset counter whenever flag is False
            groups = s.ne(s.shift()).cumsum()
            streak_vals = s.groupby(groups).cumsum()
            streaks.append(streak_vals)
        daily[streak_col] = pd.concat(streaks)

    # ── Any-anomaly flag (at least one dimension flagged) ──────────────
    anomaly_cols = [c for c in daily.columns if c.endswith("_anomaly")]
    if anomaly_cols:
        daily["any_anomaly"] = daily[anomaly_cols].any(axis=1)

    log.info("Anomaly detection complete (threshold=%.1fσ, window=%dd).",
             z_threshold, rolling_window)
    return daily


# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERATE REPORTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_cow_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Produce a cow-level summary with anomaly counts and averages."""

    agg = {
        "days_observed": ("milk_rate", "count"),
        "avg_milk_rate": ("milk_rate", "mean"),
        "std_milk_rate": ("milk_rate", "std"),
        "avg_health_score": ("health_score", "mean"),
        "max_health_score": ("health_score", "max"),
    }

    # Dynamically add sum columns for all anomaly flags
    for col in daily.columns:
        if col.endswith("_anomaly"):
            agg[col.replace("_anomaly", "_anomaly_days")] = (col, "sum")

    # Add max streak columns
    for col in daily.columns:
        if col.endswith("_streak"):
            agg["max_" + col] = (col, "max")

    if "ketosis_risk" in daily.columns:
        agg["ketosis_risk_days"] = ("ketosis_risk", "sum")

    if "any_anomaly" in daily.columns:
        agg["any_anomaly_days"] = ("any_anomaly", "sum")

    summary = (
        daily.groupby("cow_id")
        .agg(**agg)
        .reset_index()
        .sort_values("avg_health_score", ascending=False)
    )

    log.info("Cow summary generated for %d animals.", len(summary))
    return summary


def generate_alert_report(
    daily: pd.DataFrame,
    cow_summary: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
) -> None:
    """Write a Markdown alert report highlighting the most concerning cows."""

    report_date = daily["date"].max()
    # Get rows for the most recent date
    latest = daily[daily["date"] == report_date].copy()

    lines = [
        f"# 🐄 Daily Milk Anomaly Alert Report",
        f"**Report date:** {report_date:%Y-%m-%d}  ",
        f"**Cows monitored:** {daily['cow_id'].nunique()}  ",
        f"**Anomaly threshold:** Z-score based rolling window  ",
        "",
        "---",
        "",
    ]

    # ── Today's alerts ──────────────────────────────────────────────────
    if "any_anomaly" in latest.columns:
        today_alerts = latest[latest["any_anomaly"]].sort_values("health_score", ascending=False)
    else:
        today_alerts = pd.DataFrame()

    lines.append(f"## ⚠️ Today's Alerts ({len(today_alerts)} cows)")
    lines.append("")

    if len(today_alerts) == 0:
        lines.append("✅ No anomalies detected today.")
        lines.append("")
    else:
        lines.append("| Cow | Health Score | Milk Z | Fat Z | Blood | F:P | DIM | Milk Streak | Details |")
        lines.append("|-----|------------|--------|-------|-------|-----|-----|-------------|---------|")
        for _, row in today_alerts.iterrows():
            details = []
            if row.get("low_milk", False):
                details.append("⬇️ Low milk")
            if row.get("high_milk", False):
                details.append("⬆️ High milk")
            if row.get("fat_anomaly", False):
                details.append("🧈 Fat anomaly")
            if row.get("blood_serious", False):
                details.append("🩸🩸 Blood SERIOUS")
            elif row.get("blood_warning", False):
                details.append("🩸 Blood warning")
            if row.get("fpr_serious", False):
                details.append("⚡⚡ Ketosis SERIOUS")
            elif row.get("fpr_warning", False):
                details.append("⚡ Ketosis warning")
            if row.get("rumination_anomaly", False):
                details.append("🔇 Low rumination")

            dim_val = f"{int(row['dim'])}" if pd.notna(row.get("dim")) else "—"
            milk_streak = int(row.get("milk_streak", 0))
            blood_val = f"{row.get('blood', 0):.2f}"
            fpr_val = f"{row.get('fat_protein_ratio', 0):.2f}"

            lines.append(
                f"| {int(row['cow_id'])} "
                f"| {row.get('health_score', 0):.2f} "
                f"| {row.get('milk_z', 0):+.2f} "
                f"| {row.get('fat_z', 0):+.2f} "
                f"| {blood_val} "
                f"| {fpr_val} "
                f"| {dim_val} "
                f"| {milk_streak}d "
                f"| {', '.join(details) or '—'} |"
            )
        lines.append("")

    # ── Top concern cows (overall period) ──────────────────────────────
    lines.append(f"## 📊 Top {top_n} Concern Cows (Overall Period)")
    lines.append("")
    lines.append("| Rank | Cow | Avg Health Score | Anomaly Days | Max Milk Streak | Obs Days |")
    lines.append("|------|-----|-----------------|--------------|-----------------|----------|")

    for i, (_, row) in enumerate(cow_summary.head(top_n).iterrows(), 1):
        any_anom = int(row.get("any_anomaly_days", 0))
        max_streak = int(row.get("max_milk_streak", 0))
        lines.append(
            f"| {i} | {int(row['cow_id'])} "
            f"| {row['avg_health_score']:.2f} "
            f"| {any_anom} "
            f"| {max_streak}d "
            f"| {int(row['days_observed'])} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated at {datetime.now():%Y-%m-%d %H:%M:%S}*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Alert report saved to %s", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TREND VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_trend_plots(
    daily: pd.DataFrame,
    cow_summary: pd.DataFrame,
    output_dir: str,
    top_n: int = 10,
) -> None:
    """Generate per-cow trend plots for the top-N most concerning cows."""

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    top_cows = cow_summary.head(top_n)["cow_id"].tolist()
    log.info("Generating trend plots for top %d cows …", len(top_cows))

    for cow_id in top_cows:
        cow_df = daily[daily["cow_id"] == cow_id].sort_values("date")
        if len(cow_df) < 3:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Cow {int(cow_id)} — Trend Overview", fontsize=14, fontweight="bold")

        # Panel 1: Milk rate + anomaly markers
        ax1 = axes[0]
        ax1.plot(cow_df["date"], cow_df["milk_rate"], "o-", ms=3, lw=1, color="#2563EB", label="Milk rate (kg/h)")
        if "milk_anomaly" in cow_df.columns:
            anom = cow_df[cow_df["milk_anomaly"]]
            ax1.scatter(anom["date"], anom["milk_rate"], c="red", s=40, zorder=5, label="Milk anomaly")
        ax1.set_ylabel("kg / hour")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Fat % and F:P ratio
        ax2 = axes[1]
        if "fat_pct" in cow_df.columns:
            ax2.plot(cow_df["date"], cow_df["fat_pct"], "s-", ms=3, lw=1, color="#D97706", label="Fat %")
        if "fat_protein_ratio" in cow_df.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(cow_df["date"], cow_df["fat_protein_ratio"], "^-", ms=3, lw=1, color="#7C3AED", label="F:P ratio", alpha=0.7)
            ax2_twin.axhline(1.5, color="#7C3AED", ls="--", lw=0.8, alpha=0.5, label="Ketosis threshold (1.5)")
            ax2_twin.set_ylabel("F:P ratio", color="#7C3AED")
            ax2_twin.legend(loc="upper left", fontsize=8)
        ax2.set_ylabel("Fat %", color="#D97706")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Health score
        ax3 = axes[2]
        if "health_score" in cow_df.columns:
            ax3.fill_between(cow_df["date"], cow_df["health_score"], alpha=0.3, color="#DC2626")
            ax3.plot(cow_df["date"], cow_df["health_score"], "-", lw=1.5, color="#DC2626", label="Health score")
            ax3.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax3.set_ylabel("Score")
        ax3.set_xlabel("Date")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"cow_{int(cow_id)}_trend.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("Trend plots saved to %s/", plots_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_milk_anomaly_system(
    input_csv: str,
    output_dir: str = "output",
    rumination_path: str | None = None,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS,
    composite_weights: dict | None = None,
    top_n_plots: int = 10,
    top_n_alerts: int = 15,
) -> None:
    """Run the full anomaly detection pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load & clean milking data
    df = load_and_clean(input_csv)

    # 2. Load rumination data (optional)
    rumination = load_rumination(rumination_path)

    # 3. Aggregate to daily features
    daily = compute_daily_features(df, rumination)

    # 4. Detect anomalies
    daily = detect_anomalies(
        daily,
        z_threshold=z_threshold,
        rolling_window=rolling_window,
        min_window_days=min_window_days,
        composite_weights=composite_weights,
    )

    # 5. Cow-level summary
    cow_summary = generate_cow_summary(daily)

    # 6. Save CSVs
    daily.to_csv(os.path.join(output_dir, "daily_milk_anomalies.csv"), index=False)
    cow_summary.to_csv(os.path.join(output_dir, "problem_cows_summary.csv"), index=False)
    log.info("CSVs saved to %s/", output_dir)

    # 7. Alert report
    generate_alert_report(
        daily, cow_summary,
        output_path=os.path.join(output_dir, "daily_alert_report.md"),
        top_n=top_n_alerts,
    )

    # 8. Trend plots
    generate_trend_plots(daily, cow_summary, output_dir, top_n=top_n_plots)

    log.info("✅ Milk anomaly analysis completed. All results in %s/", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    rumination_dir = "rumination data"

    # ── Find input milking CSV ──────────────────────────────────────────
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]

    if len(csv_files) == 0:
        log.error("❌ No CSV file found in '%s/' folder.", input_dir)
        log.error("Please place ONE robot CSV file in the input folder.")
    elif len(csv_files) > 1:
        log.error("❌ Multiple CSV files found in '%s/':", input_dir)
        for f in csv_files:
            log.error("  - %s", f)
        log.error("Please keep only ONE CSV file in the input folder.")
    else:
        input_csv = os.path.join(input_dir, csv_files[0])
        log.info("📄 Using input file: %s", csv_files[0])

        # ── Find rumination file (optional) ─────────────────────────────
        rumination_path = None
        if os.path.isdir(rumination_dir):
            rum_files = [
                f for f in os.listdir(rumination_dir)
                if f.lower().endswith((".xlsx", ".xls"))
            ]
            if rum_files:
                rumination_path = os.path.join(rumination_dir, rum_files[0])
                log.info("📊 Using rumination file: %s", rum_files[0])

        run_milk_anomaly_system(
            input_csv=input_csv,
            output_dir=output_dir,
            rumination_path=rumination_path,
        )
