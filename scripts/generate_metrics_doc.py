# generate_metrics_doc.py
# Generates STARS Monitoring Metrics.xlsx
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from pathlib import Path

# ── Definitions sheet ─────────────────────────────────────────────────────────

DEFINITIONS = [
    ("phase",                "Pipeline maturity phase. 1 = Stability & Safety (current). 2 = Predictive Health Intelligence. 3 = Structural & Research Signals."),
    ("family",               "Logical grouping within a phase: stability | truthfulness | abundance | regularity | summary."),
    ("cadence",              "How often the metric is computed and stored: daily | weekly | monthly."),
    ("name",                 "Snake-case identifier used as metric_name in the long-format output table."),
    ("metric_display_name",  "Human-readable label for dashboards and reports."),
    ("description",          "What the metric measures and why it matters."),
    ("target",               "Unit of analysis. All STARS Phase 1 metrics are computed at the feature_segment grain."),
    ("window",               "Which time window(s) are used: train | recent | train vs recent."),
    ("grain",                "Resolution of the raw input data consumed by this metric."),
    ("value_type",           "Python/pandas dtype of the metric_value column: float | bool | int."),
    ("units",                "Interpretation of the numeric value (e.g., 'KS statistic 0-1', 'Cohen\\'s d', 'proportion')."),
    ("compute_layer",        "Where the computation happens: Python (stars_pipeline), SQL, or both."),
    ("source_table_or_asset","Python module path or SQL asset that produces this metric."),
    ("join_keys_required",   "Columns needed to join metric output to source data."),
    ("min_pairs_or_obs",     "Minimum observations required in each window before the metric is computed; returns NaN otherwise."),
    ("default_threshold",    "Canonical threshold value from MonitorConfig. 'N/A' for intermediates and summaries."),
    ("threshold_direction",  "Direction of the threshold gate: >= | <= | < | > | bidirectional | composite. 'N/A' for intermediates."),
    ("flag_logic",           "Full boolean expression that sets metric_flag=True. 'N/A' for intermediates and non-flag summaries."),
    ("default_action",       "Recommended action when metric_flag=True."),
    ("notes",                "Wiring guidance, caveats, and cross-references to MonitorConfig fields."),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

JOIN_KEYS = "strata_id, entity_id, patient_type_rollup, service_line"
COMPUTE   = "Python"
SOURCE_PFX = "stars_pipeline.stars.tests"
CADENCE   = "daily"
GRAIN     = "daily"
PHASE     = 1

def primary(
    family, name, display, description, window, value_type, units, source_fn,
    min_obs, threshold, threshold_dir, flag_logic, action, notes
):
    return {
        "phase": PHASE, "family": family, "cadence": CADENCE,
        "name": name, "metric_display_name": display,
        "description": description, "target": "feature_segment",
        "window": window, "grain": GRAIN,
        "value_type": value_type, "units": units,
        "compute_layer": COMPUTE,
        "source_table_or_asset": f"{SOURCE_PFX}.{source_fn}",
        "join_keys_required": JOIN_KEYS,
        "min_pairs_or_obs": min_obs,
        "default_threshold": threshold,
        "threshold_direction": threshold_dir,
        "flag_logic": flag_logic,
        "default_action": action,
        "notes": notes,
    }

def intermediate(family, name, display, description, window, value_type, units, source_fn, notes=""):
    return {
        "phase": PHASE, "family": family, "cadence": CADENCE,
        "name": name, "metric_display_name": display,
        "description": description, "target": "feature_segment",
        "window": window, "grain": GRAIN,
        "value_type": value_type, "units": units,
        "compute_layer": COMPUTE,
        "source_table_or_asset": f"{SOURCE_PFX}.{source_fn}",
        "join_keys_required": JOIN_KEYS,
        "min_pairs_or_obs": "same as parent",
        "default_threshold": "N/A",
        "threshold_direction": "N/A",
        "flag_logic": "N/A — intermediate only",
        "default_action": "N/A — see parent metric",
        "notes": notes,
    }

def summary(name, display, description, value_type, units, flag_logic, notes):
    return {
        "phase": PHASE, "family": "summary", "cadence": CADENCE,
        "name": name, "metric_display_name": display,
        "description": description, "target": "feature_segment",
        "window": "derived", "grain": GRAIN,
        "value_type": value_type, "units": units,
        "compute_layer": COMPUTE,
        "source_table_or_asset": "stars_pipeline.stars.monitor.apply_thresholds",
        "join_keys_required": JOIN_KEYS,
        "min_pairs_or_obs": "N/A",
        "default_threshold": "N/A",
        "threshold_direction": "N/A",
        "flag_logic": flag_logic,
        "default_action": "Route segment for review based on which family is violated.",
        "notes": notes,
    }


# ── Metrics rows ──────────────────────────────────────────────────────────────

rows = []

# ────────────────────────────────────────────────────────────── STABILITY ──

rows.append(primary(
    family="stability",
    name="ks_distribution",
    display="KS Distributional Shift",
    description="Detects distributional shift between training and recent windows using the two-sample Kolmogorov-Smirnov test.",
    window="train vs recent",
    value_type="float",
    units="KS statistic (0–1)",
    source_fn="test_ks_distribution",
    min_obs="10 train + 10 recent",
    threshold=0.30,
    threshold_dir=">=",
    flag_logic="ks_stat >= 0.30 AND p_value < 0.05",
    action="Investigate for distribution changes in volume data or upstream data pipeline anomalies.",
    notes="Both the KS statistic gate (>= ks_d_threshold) and the p-value gate (< alpha=0.05) must pass. "
          "MonitorConfig: ks_d_threshold=0.30, alpha=0.05.",
))

rows.append(intermediate("stability", "ks_distribution__ks_p_value",
    "KS Test p-value", "Two-sample KS test p-value.",
    "train vs recent", "float", "p-value (0–1)", "test_ks_distribution",
    "Intermediate for ks_distribution. Small p = distributions are different."))

rows.append(intermediate("stability", "ks_distribution__mean_train",
    "KS Mean (Train)", "Mean of daily volumes in the training window.",
    "train", "float", "volume units", "test_ks_distribution"))

rows.append(intermediate("stability", "ks_distribution__mean_recent",
    "KS Mean (Recent)", "Mean of daily volumes in the recent window.",
    "recent", "float", "volume units", "test_ks_distribution"))

# ── level_shift ──

rows.append(primary(
    family="stability",
    name="level_shift",
    display="Level Shift (Cohen's d)",
    description="Detects a statistically significant and practically large shift in mean volume using Welch's t-test with Cohen's d magnitude gate.",
    window="train vs recent",
    value_type="float",
    units="Cohen's d",
    source_fn="test_level_shift",
    min_obs="2 train + 2 recent",
    threshold=1.15,
    threshold_dir=">=",
    flag_logic="welch_t_p < 0.05 AND cohens_d >= 1.15",
    action="Investigate for step-change in mean volume. May indicate data pull changes, provider behavior shifts, or patient population changes.",
    notes="value = 0.0 when t-test p >= 0.05 (effect size not computed). Requires both statistical (Welch's t) and practical significance (Cohen's d >= 1.15 = large effect). MonitorConfig: level_shift_min_cohen_d=1.15, alpha=0.05.",
))

rows.append(intermediate("stability", "level_shift__p_value",
    "Level Shift p-value", "Welch's t-test p-value for mean difference.",
    "train vs recent", "float", "p-value (0–1)", "test_level_shift",
    "Unequal-variance t-test. Gated before Cohen's d is evaluated."))

rows.append(intermediate("stability", "level_shift__mean_train",
    "Level Shift Mean (Train)", "Mean daily volume in training window.",
    "train", "float", "volume units", "test_level_shift"))

rows.append(intermediate("stability", "level_shift__mean_recent",
    "Level Shift Mean (Recent)", "Mean daily volume in recent window.",
    "recent", "float", "volume units", "test_level_shift"))

# ── dw_shift ──

rows.append(primary(
    family="stability",
    name="dw_shift",
    display="Durbin-Watson Residual Shift",
    description="Detects shift in residual autocorrelation structure by fitting OLS on each window and comparing Durbin-Watson statistics.",
    window="train vs recent",
    value_type="float",
    units="|DW_recent - DW_train|",
    source_fn="test_dw_shift",
    min_obs="3 train + 3 recent",
    threshold=1.50,
    threshold_dir=">=",
    flag_logic="|dw_recent - dw_train| >= 1.50",
    action="Investigate for changes in serial correlation of residuals. May indicate seasonality onset, autocorrelated errors, or structural breaks.",
    notes="DW range [0, 4]: 2.0 = no autocorrelation; < 2 = positive; > 2 = negative. MonitorConfig: dw_delta_threshold=1.50.",
))

rows.append(intermediate("stability", "dw_shift__dw_train",
    "DW Statistic (Train)", "Durbin-Watson statistic computed on OLS residuals of the training window.",
    "train", "float", "DW statistic (0–4)", "test_dw_shift",
    "OLS fit: y ~ 1 + t (linear time index). DW applied to residuals."))

rows.append(intermediate("stability", "dw_shift__dw_recent",
    "DW Statistic (Recent)", "Durbin-Watson statistic computed on OLS residuals of the recent window.",
    "recent", "float", "DW statistic (0–4)", "test_dw_shift"))

# ── trend_change ──

rows.append(primary(
    family="stability",
    name="trend_change",
    display="Trend Change (Interaction OLS)",
    description="Detects a statistically significant change in trend slope between training and recent windows using an interaction-term OLS model.",
    window="train vs recent",
    value_type="float",
    units="p-value of interaction term",
    source_fn="test_trend_change",
    min_obs="10 train + 10 recent",
    threshold=0.20,
    threshold_dir="< (composite)",
    flag_logic="p_interaction < 0.20 AND max(|slope_train|, |slope_recent|) >= 0.015 AND slope_change_ratio >= 0.75",
    action="Investigate for trend emergence, reversal, or acceleration. May indicate changing patient volume trends or operational changes.",
    notes="Three-gate test: (1) p < trend_p_value_threshold=0.20, (2) max slope >= slope_threshold=0.015, (3) slope_change_ratio >= slope_change_ratio_threshold=0.75. "
          "Model: y ~ 1 + t + recent_ind + (t × recent_ind); interaction coefficient = slope delta. "
          "slope_change_ratio = |slope_delta| / (max(|slope_train|, |slope_recent|) + 1e-9).",
))

rows.append(intermediate("stability", "trend_change__slope_train",
    "Trend Slope (Train)", "OLS slope coefficient for the training window (daily volume change per day).",
    "train", "float", "volume / day", "test_trend_change",
    "Extracted from interaction OLS: params[1] = baseline slope."))

rows.append(intermediate("stability", "trend_change__slope_recent",
    "Trend Slope (Recent)", "OLS slope for the recent window = slope_train + slope_delta.",
    "recent", "float", "volume / day", "test_trend_change"))

rows.append(intermediate("stability", "trend_change__slope_delta",
    "Slope Delta", "Difference in slope: slope_recent - slope_train. Interaction coefficient from OLS.",
    "train vs recent", "float", "volume / day", "test_trend_change",
    "OLS params[3] = interaction term coefficient."))

rows.append(intermediate("stability", "trend_change__slope_change_ratio",
    "Slope Change Ratio", "Relative slope change: |slope_delta| / (max(|slope_train|, |slope_recent|) + eps).",
    "train vs recent", "float", "ratio (≥ 0)", "test_trend_change",
    "Denominator uses max of absolute slopes to prevent epsilon inflation when train slope is near zero."))

# ── stationarity ──

rows.append(primary(
    family="stability",
    name="stationarity",
    display="Stationarity Loss (KPSS)",
    description="Detects transition from stationary to non-stationary behaviour using the KPSS test on both windows.",
    window="train vs recent",
    value_type="float",
    units="KPSS statistic (recent window)",
    source_fn="test_stationarity",
    min_obs="30 train + 30 recent",
    threshold=0.10,
    threshold_dir="KPSS p-value <=",
    flag_logic="kpss_p_train > 0.10 (train is stationary) AND kpss_p_recent <= 0.10 (recent is non-stationary)",
    action="Investigate for trend onset, structural break, or seasonality changes in the recent window.",
    notes="statsmodels KPSS p-value is capped at 0.10; strict > comparison used (not >=) so p=0.10 is treated as non-stationary boundary. "
          "One-directional: only fires when training IS stationary but recent is NOT. MonitorConfig: kpss_alpha=0.10.",
))

rows.append(intermediate("stability", "stationarity__kpss_p_train",
    "KPSS p-value (Train)", "KPSS p-value for the training window (capped at 0.10 by statsmodels).",
    "train", "float", "p-value (0–0.10)", "test_stationarity",
    "p > kpss_alpha means training window is stationary."))

rows.append(intermediate("stability", "stationarity__kpss_p_recent",
    "KPSS p-value (Recent)", "KPSS p-value for the recent window (capped at 0.10 by statsmodels).",
    "recent", "float", "p-value (0–0.10)", "test_stationarity",
    "p <= kpss_alpha means recent window is non-stationary."))

rows.append(intermediate("stability", "stationarity__train_stationary",
    "Train Stationary", "True if kpss_p_train > kpss_alpha. Pre-condition for the flag.",
    "train", "bool", "True/False", "test_stationarity"))

# ────────────────────────────────────────────────────────── TRUTHFULNESS ──

rows.append(primary(
    family="truthfulness",
    name="coverage_shift",
    display="Coverage Shift",
    description="Detects a significant shift in the proportion of observed (non-missing) days between training and recent windows.",
    window="train vs recent",
    value_type="float",
    units="|coverage_recent - coverage_train| (proportion)",
    source_fn="test_coverage_shift",
    min_obs="1 train + 1 recent",
    threshold=0.40,
    threshold_dir=">=",
    flag_logic="two_prop_z_p < 0.05 AND |coverage_recent - coverage_train| >= 0.40",
    action="Investigate for data pipeline gaps, ETL failures, or changes in reporting completeness.",
    notes="Coverage = observed_days / calendar_days_in_window. Requires both statistical (z-test) and practical (delta >= 0.40) significance. MonitorConfig: coverage_delta_threshold=0.40, alpha=0.05.",
))

rows.append(intermediate("truthfulness", "coverage_shift__p_value",
    "Coverage Shift p-value", "Two-proportion z-test p-value for coverage rate difference.",
    "train vs recent", "float", "p-value (0–1)", "test_coverage_shift"))

rows.append(intermediate("truthfulness", "coverage_shift__coverage_train",
    "Coverage Rate (Train)", "Proportion of calendar days with an observed value in the training window.",
    "train", "float", "proportion (0–1)", "test_coverage_shift"))

rows.append(intermediate("truthfulness", "coverage_shift__coverage_recent",
    "Coverage Rate (Recent)", "Proportion of calendar days with an observed value in the recent window.",
    "recent", "float", "proportion (0–1)", "test_coverage_shift"))

# ── sparsity_change ──

rows.append(primary(
    family="truthfulness",
    name="sparsity_change",
    display="Sparsity Change",
    description="Detects a significant shift in the proportion of zero-value days among observed days between training and recent windows.",
    window="train vs recent",
    value_type="float",
    units="|sparsity_recent - sparsity_train| (proportion)",
    source_fn="test_sparsity_change",
    min_obs="1 train + 1 recent",
    threshold=0.30,
    threshold_dir=">=",
    flag_logic="two_prop_z_p < 0.05 AND |sparsity_recent - sparsity_train| >= 0.30",
    action="Investigate for changes in zero-reporting behavior, data suppression, or volume collapse.",
    notes="Denominator is observed days only (not calendar days). Input arrays (train_zero, recent_zero) contain only days with an observation. MonitorConfig: sparsity_delta_threshold=0.30, alpha=0.05.",
))

rows.append(intermediate("truthfulness", "sparsity_change__p_value",
    "Sparsity Change p-value", "Two-proportion z-test p-value for sparsity rate difference.",
    "train vs recent", "float", "p-value (0–1)", "test_sparsity_change"))

rows.append(intermediate("truthfulness", "sparsity_change__sparsity_train",
    "Sparsity Rate (Train)", "Proportion of observed days with a zero value in the training window.",
    "train", "float", "proportion (0–1)", "test_sparsity_change"))

rows.append(intermediate("truthfulness", "sparsity_change__sparsity_recent",
    "Sparsity Rate (Recent)", "Proportion of observed days with a zero value in the recent window.",
    "recent", "float", "proportion (0–1)", "test_sparsity_change"))

# ───────────────────────────────────────────────────────────── ABUNDANCE ──

rows.append(primary(
    family="abundance",
    name="low_volume",
    display="Low Volume Flag",
    description="Detects insufficient signal volume by checking whether mean monthly volume in the training window falls below threshold.",
    window="train",
    value_type="float",
    units="avg monthly volume",
    source_fn="test_low_volume",
    min_obs="1 train day",
    threshold=3.0,
    threshold_dir="<",
    flag_logic="avg_monthly_volume_train < 3.0",
    action="Flag segment for exclusion from statistical monitoring; insufficient data for reliable inference.",
    notes="DISABLED by default: low_volume_enabled=False in MonitorConfig. Statistics are computed and stored but do not contribute to abundance_violations or is_flagged unless low_volume_enabled=True. "
          "Monthly grouping uses calendar months (pd.Period). MonitorConfig: low_volume_monthly_threshold=3.0.",
))

rows.append(intermediate("abundance", "low_volume__total_volume_train",
    "Total Volume (Train)", "Sum of all daily volumes in the training window.",
    "train", "float", "volume units", "test_low_volume"))

rows.append(intermediate("abundance", "low_volume__n_months_train",
    "Number of Months (Train)", "Count of distinct calendar months in the training window.",
    "train", "int", "count", "test_low_volume"))

# ───────────────────────────────────────────────────────────── REGULARITY ──

rows.append(primary(
    family="regularity",
    name="volatility_shift",
    display="Volatility Shift (CV Ratio)",
    description="Detects a disproportionate shift in relative variability using the coefficient of variation ratio (CV_recent / CV_train).",
    window="train vs recent",
    value_type="float",
    units="CV ratio (CV_recent / CV_train)",
    source_fn="test_volatility_shift",
    min_obs="10 train + 10 recent",
    threshold=2.75,
    threshold_dir="bidirectional",
    flag_logic="cv_ratio >= 2.75 OR cv_ratio <= 1/2.75 (~0.364)",
    action="Investigate for changes in variability pattern — either volatility explosion or suppression.",
    notes="DISABLED by default: volatility_shift_enabled=False in MonitorConfig. Statistics are computed and stored but do not contribute to regularity_violations or is_flagged unless volatility_shift_enabled=True. "
          "CV = std / |mean|. Returns NaN when mean < 1e-6. MonitorConfig: volatility_ratio_threshold=2.75.",
))

rows.append(intermediate("regularity", "volatility_shift__cv_train",
    "CV (Train)", "Coefficient of variation for the training window: std / |mean|.",
    "train", "float", "ratio (≥ 0)", "test_volatility_shift"))

rows.append(intermediate("regularity", "volatility_shift__cv_recent",
    "CV (Recent)", "Coefficient of variation for the recent window: std / |mean|.",
    "recent", "float", "ratio (≥ 0)", "test_volatility_shift"))

rows.append(intermediate("regularity", "volatility_shift__mean_train",
    "Volatility Mean (Train)", "Mean daily volume in training window (used for CV denominator).",
    "train", "float", "volume units", "test_volatility_shift"))

rows.append(intermediate("regularity", "volatility_shift__mean_recent",
    "Volatility Mean (Recent)", "Mean daily volume in recent window (used for CV denominator).",
    "recent", "float", "volume units", "test_volatility_shift"))

# ── outlier_rate ──

rows.append(primary(
    family="regularity",
    name="outlier_rate",
    display="Outlier Rate",
    description="Detects excessive proportion of outlier points in the recent window using a modified z-score baseline from the training window.",
    window="train baseline + recent evaluation",
    value_type="float",
    units="proportion (outlier_count / len(recent))",
    source_fn="test_outlier_rate",
    min_obs="3 train + 1 recent",
    threshold=0.40,
    threshold_dir=">=",
    flag_logic="(outlier_count / len(recent)) >= 0.40",
    action="Investigate for outlier events, data anomalies, or extreme values in the recent window.",
    notes="Modified z-score: |value - train_median| / (train_mad / 0.6745). Outlier if z > outlier_z_threshold=3.50. "
          "When train_mad=0, falls back to train_std. train_median and train_mad define the baseline — only recent values are evaluated against it. "
          "MonitorConfig: outlier_z_threshold=3.50, outlier_rate_threshold=0.40.",
))

rows.append(intermediate("regularity", "outlier_rate__outlier_count",
    "Outlier Count", "Number of outlier points found in the recent window.",
    "recent", "int", "count", "test_outlier_rate",
    "Points where |value - train_median| / scale > outlier_z_threshold=3.50."))

rows.append(intermediate("regularity", "outlier_rate__train_median",
    "Train Median", "Median of the training window used as the outlier baseline.",
    "train", "float", "volume units", "test_outlier_rate"))

rows.append(intermediate("regularity", "outlier_rate__train_mad",
    "Train MAD", "Median absolute deviation of the training window: median(|x - median(x)|).",
    "train", "float", "volume units", "test_outlier_rate",
    "When MAD=0, outlier scale uses std/0.6745 instead."))

# ── acf_structure ──

rows.append(primary(
    family="regularity",
    name="acf_structure",
    display="ACF Structure Divergence",
    description="Detects divergence in autocorrelation structure at lags 1, 7, and 30 using Fisher Z-transform test — only for lags significant in training.",
    window="train vs recent",
    value_type="float",
    units="min Fisher Z p-value across tested lags (NaN if no lags pass Bartlett gate)",
    source_fn="test_acf_structure",
    min_obs="10 train + 10 recent",
    threshold=0.05,
    threshold_dir="< (any lag)",
    flag_logic="any lag where |acf_train_lagN| > bartlett_bound AND fisher_z_p < 0.05",
    action="Investigate for changes in temporal patterns. Lag 1 = daily carryover; lag 7 = weekly cycle; lag 30 = monthly pattern.",
    notes="Bartlett bound = 1.96 / sqrt(n_train). Lags where training ACF does not exceed the Bartlett bound are not tested. "
          "Fisher Z-transform: arctanh(r), SE = sqrt(1/(n-3)). Value = min p-value across tested lags. "
          "MonitorConfig: acf_divergence_p_threshold=0.05.",
))

rows.append(intermediate("regularity", "acf_structure__bartlett_bound",
    "Bartlett Bound", "Significance threshold for training ACF lags: 1.96 / sqrt(n_train).",
    "train", "float", "ACF magnitude", "test_acf_structure",
    "Gate: lag is only tested if |acf_train_lagN| > bartlett_bound."))

rows.append(intermediate("regularity", "acf_structure__acf_train_lag1",
    "ACF Lag 1 (Train)", "Autocorrelation at lag 1 for the training window.",
    "train", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_recent_lag1",
    "ACF Lag 1 (Recent)", "Autocorrelation at lag 1 for the recent window.",
    "recent", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_p_lag1",
    "ACF Fisher Z p-value (Lag 1)", "Fisher Z-transform test p-value comparing lag-1 ACF between windows.",
    "train vs recent", "float", "p-value (0–1)", "test_acf_structure",
    "NaN if lag 1 training ACF does not exceed bartlett_bound."))

rows.append(intermediate("regularity", "acf_structure__acf_train_lag7",
    "ACF Lag 7 (Train)", "Autocorrelation at lag 7 (weekly pattern) for the training window.",
    "train", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_recent_lag7",
    "ACF Lag 7 (Recent)", "Autocorrelation at lag 7 (weekly pattern) for the recent window.",
    "recent", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_p_lag7",
    "ACF Fisher Z p-value (Lag 7)", "Fisher Z-transform test p-value comparing lag-7 ACF between windows.",
    "train vs recent", "float", "p-value (0–1)", "test_acf_structure",
    "NaN if lag 7 training ACF does not exceed bartlett_bound. Lag 7 = weekly cycle detection."))

rows.append(intermediate("regularity", "acf_structure__acf_train_lag30",
    "ACF Lag 30 (Train)", "Autocorrelation at lag 30 (monthly pattern) for the training window.",
    "train", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_recent_lag30",
    "ACF Lag 30 (Recent)", "Autocorrelation at lag 30 (monthly pattern) for the recent window.",
    "recent", "float", "correlation (-1 to 1)", "test_acf_structure"))

rows.append(intermediate("regularity", "acf_structure__acf_p_lag30",
    "ACF Fisher Z p-value (Lag 30)", "Fisher Z-transform test p-value comparing lag-30 ACF between windows.",
    "train vs recent", "float", "p-value (0–1)", "test_acf_structure",
    "NaN if lag 30 training ACF does not exceed bartlett_bound. Lag 30 = monthly cycle detection."))

# ──────────────────────────────────────────────────────────────── SUMMARY ──

rows.append(summary(
    name="is_flagged",
    display="Is Flagged",
    description="True if any STARS family has at least one active violation for this segment.",
    value_type="bool",
    units="True / False",
    flag_logic="stability_violations > 0 OR truthfulness_violations > 0 OR abundance_violations > 0 OR regularity_violations > 0",
    notes="Computed by apply_thresholds(). Optional flags (low_volume, volatility_shift) are zeroed before counting if their *_enabled=False. "
          "stars_family='Summary' in long-format output.",
))

rows.append(summary(
    name="stability_violations",
    display="Stability Violations",
    description="Count of Stability family tests that fired (ks_distribution, level_shift, dw_shift, trend_change, stationarity).",
    value_type="int",
    units="count (0–5)",
    flag_logic="sum of: ks_distribution_flag, level_shift_flag, dw_shift_flag, trend_change_flag, stationarity_flag",
    notes="5 possible Stability tests. Maximum = 5.",
))

rows.append(summary(
    name="truthfulness_violations",
    display="Truthfulness Violations",
    description="Count of Truthfulness family tests that fired (coverage_shift, sparsity_change).",
    value_type="int",
    units="count (0–2)",
    flag_logic="sum of: coverage_shift_flag, sparsity_change_flag",
    notes="2 possible Truthfulness tests. Maximum = 2.",
))

rows.append(summary(
    name="abundance_violations",
    display="Abundance Violations",
    description="Count of Abundance family tests that fired (low_volume — disabled by default).",
    value_type="int",
    units="count (0–1)",
    flag_logic="sum of: low_volume_flag (zeroed when low_volume_enabled=False)",
    notes="1 possible Abundance test, disabled by default. Maximum = 1 when enabled.",
))

rows.append(summary(
    name="regularity_violations",
    display="Regularity Violations",
    description="Count of Regularity family tests that fired (volatility_shift disabled by default, outlier_rate, acf_structure).",
    value_type="int",
    units="count (0–3)",
    flag_logic="sum of: volatility_shift_flag (zeroed when disabled), outlier_rate_flag, acf_structure_flag",
    notes="3 possible Regularity tests (1 disabled by default). Maximum = 3 when all enabled.",
))


# ── Build DataFrames ──────────────────────────────────────────────────────────

COLS = [
    "phase", "family", "cadence", "name", "metric_display_name", "description",
    "target", "window", "grain", "value_type", "units", "compute_layer",
    "source_table_or_asset", "join_keys_required", "min_pairs_or_obs",
    "default_threshold", "threshold_direction", "flag_logic", "default_action", "notes",
]

metrics_df = pd.DataFrame(rows, columns=COLS)

defs_df = pd.DataFrame(DEFINITIONS, columns=["field_name", "definition"])

# ── Write to Excel ────────────────────────────────────────────────────────────

out_path = Path(r"C:\Users\zrhodes\Documents\Monitoring\STARS Monitoring Metrics.xlsx")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    defs_df.to_excel(writer, sheet_name="Definitions", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    # Auto-fit column widths (approx)
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

print(f"Written: {out_path}")
print(f"  Definitions: {len(defs_df)} rows")
print(f"  Metrics:     {len(metrics_df)} rows ({metrics_df['family'].value_counts().to_dict()})")
