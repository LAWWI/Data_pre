"""Feature Selection Method 3: Random Forest Feature Importance Analysis
ใช้ Random Forest Classifier (Non-linear Ensemble) คำนวณค่า Mean Decrease in Impurity (Gini Importance)
เพื่อจัดอันดับฟีเจอร์ที่มีพลังในการแยกโกง/ไม่โกงมากที่สุด
รันบนชุดข้อมูล Train เพื่อป้องกัน Data Leakage
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_NPZ = ROOT / "outputs" / "processed.npz"
FEATURE_DIR = Path(__file__).resolve().parent


def run_rf_importance_analysis():
    print("=" * 60)
    print("1. Loading training dataset from outputs/processed.npz...")
    data = np.load(PROCESSED_NPZ)
    X_train = data["X_train"]
    y_train = data["y_train"]
    features = [str(f) for f in data["features"]]

    print(f"Loaded {X_train.shape[0]:,} training rows with {X_train.shape[1]} features.")

    # Train Random Forest Classifier
    print("\n2. Training RandomForestClassifier (100 trees, balanced weights)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 2)
    print(f"Random Forest model fitted in {fit_time} seconds.")

    # Extract Feature Importances
    importances = rf.feature_importances_
    results_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances,
        "Importance_Pct": importances * 100.0
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    results_df["Rank"] = range(1, len(results_df) + 1)
    results_df["Cumulative_Pct"] = results_df["Importance_Pct"].cumsum()

    # Save to CSV
    csv_path = FEATURE_DIR / "rf_importance_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n3. Saved complete Random Forest importance table to: {csv_path.name}")

    # Determine thresholds (features needed for 80% and 90% power)
    feats_for_80 = results_df[results_df["Cumulative_Pct"] <= 80]["Feature"].tolist()
    feats_for_90 = results_df[results_df["Cumulative_Pct"] <= 90]["Feature"].tolist()

    summary = {
        "method": "Random Forest Feature Importance (Mean Decrease in Impurity)",
        "model_params": {
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced",
            "fit_time_seconds": fit_time
        },
        "dataset_split": "Train Set (80%)",
        "total_features": len(features),
        "features_for_80_pct_power": len(feats_for_80),
        "features_for_90_pct_power": len(feats_for_90),
        "top_10_features": [
            {
                "rank": int(row["Rank"]),
                "feature": row["Feature"],
                "importance_pct": round(float(row["Importance_Pct"]), 2),
                "cumulative_pct": round(float(row["Cumulative_Pct"]), 2)
            }
            for _, row in results_df.head(10).iterrows()
        ],
        "bottom_5_features": [
            {
                "rank": int(row["Rank"]),
                "feature": row["Feature"],
                "importance_pct": round(float(row["Importance_Pct"]), 3)
            }
            for _, row in results_df.tail(5).iterrows()
        ]
    }
    json_path = FEATURE_DIR / "rf_importance_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved Random Forest importance summary to: {json_path.name}")

    # Plot Bar Chart
    print("\n4. Generating feature importance bar chart...")
    plot_df = results_df.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    bars = ax.barh(plot_df["Feature"], plot_df["Importance_Pct"], color="#2563eb", height=0.65)

    ax.set_title("Top 15 Features by Random Forest Importance (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance (% of total decision power)", fontsize=10, labelpad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}%", va="center", ha="left", fontsize=8.5,
                fontweight="medium", color="#1f2937")

    ax.set_xlim(0, max(plot_df["Importance_Pct"]) * 1.18)

    img_path = FEATURE_DIR / "rf_importance_barplot.png"
    fig.savefig(img_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved plot to: {img_path.name}")

    print("=" * 60)
    print("Top 10 Most Important Features (Random Forest):")
    for idx, row in results_df.head(10).iterrows():
        print(f"  Rank {row['Rank']:2.0f}: {row['Feature']:15s} {row['Importance_Pct']:5.2f}% (Cumulative: {row['Cumulative_Pct']:5.2f}%)")
    print("=" * 60)
    print(f"Key insight: Just {len(feats_for_80)} features account for ~80% of all decision power!")
    print(f"Features: {', '.join(feats_for_80)}")
    print("=" * 60)


if __name__ == "__main__":
    run_rf_importance_analysis()
