"""Feature Selection Method 1: Correlation Matrix Analysis
วิเคราะห์สหสัมพันธ์ (Pearson Correlation) ระหว่างฟีเจอร์กับตัวแปรเป้าหมาย Class (Fraud)
รันบนชุดข้อมูล Train เพื่อป้องกัน Data Leakage
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_NPZ = ROOT / "outputs" / "processed.npz"
FEATURE_DIR = Path(__file__).resolve().parent


def run_correlation_analysis():
    print("=" * 60)
    print("1. Loading training dataset from outputs/processed.npz...")
    data = np.load(PROCESSED_NPZ)
    X_train = data["X_train"]
    y_train = data["y_train"]
    features = [str(f) for f in data["features"]]

    print(f"Loaded {X_train.shape[0]:,} training rows with {X_train.shape[1]} features.")

    # Convert to DataFrame
    df = pd.DataFrame(X_train, columns=features)
    df["Class"] = y_train

    # Calculate Pearson Correlation with Class
    print("2. Calculating Pearson correlation with Class (Fraud)...")
    corr = df.corr()["Class"].drop("Class")

    results_df = pd.DataFrame({
        "Feature": corr.index,
        "Correlation": corr.values,
        "Abs_Correlation": corr.abs().values,
        "Direction": np.where(corr.values >= 0, "Positive", "Negative")
    }).sort_values(by="Abs_Correlation", ascending=False).reset_index(drop=True)

    # Save to CSV (with utf-8-sig for Excel compatibility)
    csv_path = FEATURE_DIR / "correlation_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved complete correlation table to: {csv_path.name}")

    # Save summary to JSON
    top_15 = results_df.head(15).to_dict(orient="records")
    bottom_5 = results_df.tail(5).to_dict(orient="records")
    summary = {
        "method": "Pearson Correlation Matrix",
        "dataset_split": "Train Set (80%)",
        "rows_analyzed": int(len(df)),
        "total_features": int(len(corr)),
        "top_15_features": [
            {
                "feature": row["Feature"],
                "correlation": round(float(row["Correlation"]), 4),
                "direction": row["Direction"]
            }
            for row in top_15
        ],
        "bottom_5_features": [
            {
                "feature": row["Feature"],
                "correlation": round(float(row["Correlation"]), 4),
                "direction": row["Direction"]
            }
            for row in bottom_5
        ]
    }
    json_path = FEATURE_DIR / "correlation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved correlation summary to: {json_path.name}")

    # Plot Bar Chart
    print("3. Generating correlation bar chart...")
    top_features_df = results_df.head(15).iloc[::-1]  # reverse for top-down display

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    colors = ["#2563eb" if x >= 0 else "#dc2626" for x in top_features_df["Correlation"]]
    bars = ax.barh(top_features_df["Feature"], top_features_df["Correlation"], color=colors, height=0.65)

    ax.axvline(0, color="#4b5563", linewidth=0.8, linestyle="--")
    ax.set_title("Top 15 Features by Correlation with Fraud (Class)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=10, labelpad=8)
    ax.set_xlim(-0.40, 0.25)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ha = "right" if width < 0 else "left"
        offset = -0.012 if width < 0 else 0.012
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2,
                f"{width:+.4f}", va="center", ha=ha, fontsize=8.5,
                fontweight="medium", color="#1f2937")

    # Legend proxy
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="#dc2626")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="#2563eb")
    ax.legend([red_patch, blue_patch], ["Negative (Lower value = Higher fraud)", "Positive (Higher value = Higher fraud)"],
              loc="lower right", frameon=True, fontsize=8.5)

    img_path = FEATURE_DIR / "correlation_barplot.png"
    fig.savefig(img_path, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"Saved plot to: {img_path.name}")

    print("=" * 60)
    print("Top 10 Most Influential Features:")
    for idx, row in results_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['Feature']:15s} r = {row['Correlation']:+.4f} ({row['Direction']})")
    print("=" * 60)


if __name__ == "__main__":
    run_correlation_analysis()
