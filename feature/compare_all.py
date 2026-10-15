"""Generate short, unified comparison chart & table across all 3 feature selection methods."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_DIR = Path(__file__).resolve().parent

# Load 3 results
corr_df = pd.read_csv(FEATURE_DIR / "correlation" / "correlation_results.csv").set_index("Feature")
lasso_df = pd.read_csv(FEATURE_DIR / "lasso" / "lasso_results.csv").set_index("Feature")
rf_df = pd.read_csv(FEATURE_DIR / "random_forest" / "rf_importance_results.csv").set_index("Feature")

# Combine top features from Random Forest
top10_feats = rf_df.head(10).index.tolist()

comp_df = pd.DataFrame({
    "RF_Importance_Pct": rf_df.loc[top10_feats, "Importance_Pct"],
    "Lasso_Abs_Coef": lasso_df.loc[top10_feats, "Abs_Coefficient"],
    "Corr_Abs_r": corr_df.loc[top10_feats, "Abs_Correlation"]
}).round(4)

comp_df.to_csv(FEATURE_DIR / "feature_comparison_top10.csv", encoding="utf-8-sig")

# Plot 3-panel quick comparison
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=150)
plot_feats = top10_feats[::-1]

# Panel 1: Correlation
axes[0].barh(plot_feats, [corr_df.loc[f, "Abs_Correlation"] for f in plot_feats], color="#6366f1")
axes[0].set_title("1) Correlation (|r|)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("|Pearson r|")

# Panel 2: Lasso
axes[1].barh(plot_feats, [lasso_df.loc[f, "Abs_Coefficient"] for f in plot_feats], color="#ec4899")
axes[1].set_title("2) Lasso (|Coefficient|)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("|L1 Weight|")

# Panel 3: Random Forest
axes[2].barh(plot_feats, [rf_df.loc[f, "Importance_Pct"] for f in plot_feats], color="#10b981")
axes[2].set_title("3) Random Forest (%)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Importance %")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
fig.savefig(FEATURE_DIR / "feature_comparison_summary.png", bbox_inches="tight", facecolor="#ffffff")
plt.close(fig)
print("Saved feature_comparison_summary.png and feature_comparison_top10.csv")
