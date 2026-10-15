"""Feature Selection Method 2: L1 Regularization (Lasso) Analysis
ใช้ Logistic Regression ร่วมกับ L1 penalty (Lasso) บีบสัมประสิทธิ์ของฟีเจอร์ที่ไม่จำเป็นให้เป็น 0
รันบนชุดข้อมูล Train เท่านั้น เพื่อป้องกัน Data Leakage
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_NPZ = ROOT / "outputs" / "processed.npz"
FEATURE_DIR = Path(__file__).resolve().parent


def run_lasso_analysis():
    print("=" * 60)
    print("1. Loading training dataset from outputs/processed.npz...")
    data = np.load(PROCESSED_NPZ)
    X_train = data["X_train"]
    y_train = data["y_train"]
    features = [str(f) for f in data["features"]]

    print(f"Loaded {X_train.shape[0]:,} training rows with {X_train.shape[1]} features.")

    # Compare different C values (Regularization Strength)
    print("\n2. Analyzing feature sparsity across regularization levels (C)...")
    c_levels = [1.0, 0.1, 0.01]
    c_summary = {}

    for c_val in c_levels:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=c_val, random_state=42)
        clf.fit(X_train, y_train)
        coefs = clf.coef_[0]
        selected_count = int(np.sum(coefs != 0))
        dropped_count = int(len(features) - selected_count)
        c_summary[f"C={c_val}"] = {
            "selected_features": selected_count,
            "dropped_features": dropped_count,
            "sparsity_pct": round(100 * dropped_count / len(features), 1)
        }
        print(f"  - C={c_val:<4}: Retained {selected_count:2d}/{len(features)} features (Zeroed out {dropped_count:2d} features)")

    # Primary Model: C=0.1 provides a great balance of sparsity and accuracy
    print("\n3. Training primary Lasso model (C=0.1)...")
    model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=42)
    model.fit(X_train, y_train)
    coefs = model.coef_[0]

    # Create Results DataFrame
    results_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs),
        "Status": np.where(coefs != 0, "Selected", "Eliminated (Zero)")
    }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = FEATURE_DIR / "lasso_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved complete Lasso table to: {csv_path.name}")

    # Save JSON summary
    selected_feats = results_df[results_df["Status"] == "Selected"].to_dict(orient="records")
    eliminated_feats = results_df[results_df["Status"] == "Eliminated (Zero)"]["Feature"].tolist()

    summary = {
        "method": "L1 Regularization (Lasso) via LogisticRegression",
        "solver": "liblinear",
        "primary_C": 0.1,
        "dataset_split": "Train Set (80%)",
        "total_features": len(features),
        "selected_count": len(selected_feats),
        "eliminated_count": len(eliminated_feats),
        "c_comparison": c_summary,
        "selected_features": [
            {
                "feature": row["Feature"],
                "coefficient": round(float(row["Coefficient"]), 4),
                "direction": "Positive" if row["Coefficient"] > 0 else "Negative"
            }
            for row in selected_feats
        ],
        "eliminated_features": eliminated_feats
    }
    json_path = FEATURE_DIR / "lasso_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Lasso summary to: {json_path.name}")

    # Plot Bar Chart
    print("\n4. Generating Lasso coefficient bar chart...")
    # Plot top 20 non-zero features
    plot_df = results_df.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    colors = ["#2563eb" if x > 0 else ("#dc2626" if x < 0 else "#9ca3af") for x in plot_df["Coefficient"]]
    bars = ax.barh(plot_df["Feature"], plot_df["Coefficient"], color=colors, height=0.65)

    ax.axvline(0, color="#4b5563", linewidth=0.8, linestyle="--")
    ax.set_title("L1 Regularization (Lasso C=0.1): Feature Coefficients", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Lasso Coefficient Value", fontsize=10, labelpad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text labels
    for bar in bars:
        width = bar.get_width()
        ha = "right" if width < 0 else "left"
        offset = -0.03 if width < 0 else 0.03
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2,
                f"{width:+.3f}", va="center", ha=ha, fontsize=8.5,
                fontweight="medium", color="#1f2937")

    red_patch = plt.Rectangle((0, 0), 1, 1, fc="#dc2626")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="#2563eb")
    ax.legend([red_patch, blue_patch],
              ["Negative Weight (Low value -> Fraud)", "Positive Weight (High value -> Fraud)"],
              loc="lower right", frameon=True, fontsize=8.5)

    img_path = FEATURE_DIR / "lasso_coefficients.png"
    fig.savefig(img_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved plot to: {img_path.name}")

    print("=" * 60)
    print(f"Summary: Lasso kept {len(selected_feats)} features and eliminated {len(eliminated_feats)} features.")
    print("Top 5 Positive Signals (Higher value -> Fraud):")
    pos_top = [r for r in selected_feats if r["Coefficient"] > 0][:5]
    for r in pos_top:
        print(f"  + {r['Feature']:15s} coef = {r['Coefficient']:+.4f}")

    print("\nTop 5 Negative Signals (Lower value -> Fraud):")
    neg_top = [r for r in selected_feats if r["Coefficient"] < 0][:5]
    for r in neg_top:
        print(f"  - {r['Feature']:15s} coef = {r['Coefficient']:+.4f}")

    print(f"\nEliminated Features (Weights shrunk to exactly 0):")
    print("  " + ", ".join(eliminated_feats))
    print("=" * 60)


if __name__ == "__main__":
    run_lasso_analysis()
