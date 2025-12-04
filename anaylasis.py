# analysis.py
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict
from models import BuildingManager
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ingest_csv_folder(data_dir: str = "data") -> pd.DataFrame:
    """
    Read all CSVs from data_dir, return a cleaned combined DataFrame with columns:
    timestamp (datetime), kWh (float), Building (string)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    files = list(data_dir.glob("*.csv"))
    if not files:
        logging.warning("No CSV files found in data directory.")
        return pd.DataFrame(columns=["timestamp", "kWh", "Building"]).set_index("timestamp")

    rows = []
    bad_files = []
    for f in files:
        try:
            # try to read robustly: treat bad lines as NaN then drop them
            df = pd.read_csv(f)
            # infer building name if not present
            building_name = None
            if "Building" in df.columns:
                building_name = df["Building"].iloc[0]
            else:
                # use filename
                building_name = f.stem

            # Standardize column names (common variants)
            col_map = {}
            for c in df.columns:
                lc = c.strip().lower()
                if lc in ("timestamp", "time", "date"):
                    col_map[c] = "timestamp"
                if lc in ("kwh", "kw", "energy"):
                    col_map[c] = "kWh"
            df = df.rename(columns=col_map)

            # require timestamp + kWh
            if "timestamp" not in df.columns or "kWh" not in df.columns:
                logging.warning(f"Skipping {f} — missing required columns. Columns: {df.columns.tolist()}")
                bad_files.append(str(f))
                continue

            df["Building"] = building_name
            # parse timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # parse kWh to numeric (coerce errors)
            df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")
            # drop rows with NaT or NaN
            df = df.dropna(subset=["timestamp", "kWh"])

            rows.append(df[["timestamp", "kWh", "Building"]])
        except Exception as e:
            logging.exception(f"Failed to read {f}: {e}")
            bad_files.append(str(f))

    if not rows:
        logging.error("No valid data read from CSVs.")
        return pd.DataFrame(columns=["timestamp", "kWh", "Building"]).set_index("timestamp")

    combined = pd.concat(rows, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp")
    combined = combined.set_index("timestamp")
    logging.info(f"Combined dataframe created with {len(combined)} rows from {len(files)} files.")
    if bad_files:
        logging.info(f"Files skipped or had errors: {bad_files}")
    return combined

# Aggregation functions
def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame indexed by timestamp with column 'kWh' and optional 'Building' column.
    Returns daily totals (all buildings combined) as DataFrame with column 'kWh'.
    """
    if df.empty:
        return pd.DataFrame(columns=["kWh"])
    daily = df.resample("D")["kWh"].sum().to_frame()
    daily.index.name = "date"
    daily = daily.rename(columns={"kWh": "daily_kWh"})
    return daily

def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["kWh"])
    weekly = df.resample("W")["kWh"].sum().to_frame()
    weekly.index.name = "week_end"
    weekly = weekly.rename(columns={"kWh": "weekly_kWh"})
    return weekly

def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Building" not in df.columns:
        return pd.DataFrame(columns=["mean","min","max","sum"])
    grouped = df.reset_index().groupby("Building")["kWh"].agg(['mean','min','max','sum']).round(3)
    grouped = grouped.rename(columns={"mean":"mean_kWh","min":"min_kWh","max":"max_kWh","sum":"total_kWh"})
    return grouped

def find_peak_time(df: pd.DataFrame) -> pd.Series:
    """Return timestamp with maximum single kWh reading."""
    if df.empty:
        return pd.Series()

    # Find index of maximum kWh
    idx = df["kWh"].idxmax()

    # Extract the FIRST value even if multiple rows share the timestamp
    kwh_series = df.loc[idx, "kWh"]

    # If multiple values → take first
    if hasattr(kwh_series, "__iter__"):
        peak_value = float(kwh_series.iloc[0])
    else:
        peak_value = float(kwh_series)

    return pd.Series({
        "timestamp": idx,
        "kWh": peak_value
    })



def export_csvs(cleaned_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str = "output"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    cleaned_path = out / "cleaned_energy_data.csv"
    summary_path = out / "building_summary.csv"
    cleaned_df.reset_index().to_csv(cleaned_path, index=False)
    summary_df.to_csv(summary_path)
    return str(cleaned_path), str(summary_path)

from pathlib import Path