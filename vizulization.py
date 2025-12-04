# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def create_dashboard(cleaned_df: pd.DataFrame, output_path: str = "output/dashboard.png"):
    """
    Create a 3-chart dashboard:
     - Trend line (daily totals)
     - Bar chart (average weekly usage by building)
     - Scatter plot (peak hourly consumption points)
    """
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if cleaned_df.empty:
        logging.warning("Empty dataframe supplied to create_dashboard. Skipping plot.")
        return

    # Daily totals
    daily = cleaned_df.resample("D")["kWh"].sum()

    # Weekly avg per building: group by Building & week
    if "Building" in cleaned_df.columns:
        weekly_by_building = cleaned_df.reset_index().set_index("timestamp").groupby("Building")["kWh"].resample("W").mean().reset_index()
        avg_weekly = weekly_by_building.groupby("Building")["kWh"].mean().sort_values(ascending=False)
    else:
        avg_weekly = pd.Series(dtype=float)

    # Peak hourly points (top 100)
    top_points = cleaned_df.sort_values("kWh", ascending=False).head(100).reset_index()

    # create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    # 1) Trend line - daily
    axes[0].plot(daily.index, daily.values)
    axes[0].set_title("Daily Campus Electricity Consumption")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("kWh (daily total)")
    axes[0].grid(True)

    # 2) Bar Chart - average weekly usage per building
    axes[1].bar(avg_weekly.index, avg_weekly.values)
    axes[1].set_title("Average Weekly Usage by Building")
    axes[1].set_xlabel("Building")
    axes[1].set_ylabel("Average weekly kWh")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis="y")

    # 3) Scatter - peak hourly consumption (kWh vs timestamp), color by building if present
    if "Building" in top_points.columns:
        buildings = top_points["Building"].unique()
        for b in buildings:
            subset = top_points[top_points["Building"] == b]
            axes[2].scatter(subset["timestamp"], subset["kWh"], label=b, s=20)
        axes[2].legend()
    else:
        axes[2].scatter(top_points["timestamp"], top_points["kWh"], s=20)
    axes[2].set_title("Top Consumption Readings (Peak hours)")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("kWh (single reading)")
    axes[2].grid(True)

    plt.suptitle("Campus Energy Dashboard", fontsize=16)
    plt.savefig(outp, dpi=150)
    logging.info(f"Dashboard saved to {outp}")
    plt.close(fig)
    return str(outp)