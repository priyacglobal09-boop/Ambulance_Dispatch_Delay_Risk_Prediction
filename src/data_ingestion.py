import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


KAGGLE_DATASET = "datasetengineer/integrated-emergency-response-dataset-ierad"
PROJECT_COLUMNS = [
    "case_id",
    "priority_level",
    "incident_type",
    "time_of_day",
    "weather_conditions",
    "day_of_week",
    "traffic_density",
    "dispatch_zone",
    "distance_to_scene",
    "crew_experience_years",
    "ambulance_age_years",
    "temperature",
    "historical_zone_delay_rate",
    "caller_stress_score",
    "delay_risk",
]


ALIASES = {
    "case_id": ["case_id", "incident_id", "emergency_id", "id", "record_id", "call_id"],
    "priority_level": ["priority_level", "emergency_level", "incident_severity", "priority", "severity", "urgency", "incident_priority"],
    "incident_type": ["incident_type", "emergency_type", "event_type", "call_type", "incident_category"],
    "time_of_day": ["time_of_day", "period_of_day", "day_period"],
    "weather_conditions": ["weather_conditions", "weather", "weather_condition"],
    "day_of_week": ["day_of_week", "weekday_weekend", "day_type"],
    "traffic_density": ["traffic_density", "traffic_congestion", "traffic", "traffic_condition", "traffic_level"],
    "dispatch_zone": ["dispatch_zone", "region_type", "zone", "region", "area", "location", "district"],
    "distance_to_scene": ["distance_to_scene", "distance_to_incident", "distance", "distance_km", "distance_miles"],
    "crew_experience_years": ["crew_experience_years", "ambulance_speed", "drone_speed", "crew_experience", "experience_years"],
    "ambulance_age_years": ["ambulance_age_years", "fuel_level", "ambulance_age", "vehicle_age_years", "vehicle_age"],
    "temperature": ["temperature", "hospital_capacity", "battery_life", "temperature_c", "temperature_f", "temp"],
    "historical_zone_delay_rate": ["historical_zone_delay_rate", "zone_delay_rate", "historical_delay_rate"],
    "caller_stress_score": ["caller_stress_score", "number_of_injuries", "caller_stress", "stress_score", "caller_anxiety"],
    "delay_risk": ["delay_risk", "delayed", "is_delayed", "delay", "response_delay"],
    "response_time": ["response_time", "response_time_minutes", "response_time_min", "arrival_time_minutes"],
    "dispatch_time": ["dispatch_time", "dispatch_timestamp", "time_dispatched"],
    "arrival_time": ["arrival_time", "arrival_timestamp", "time_arrived", "on_scene_time"],
    "call_time": ["timestamp", "call_time", "call_timestamp", "reported_time", "incident_time"],
}


def canonicalize_column(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("%", "percent")
        .replace("#", "number")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [canonicalize_column(col) for col in normalized.columns]
    return normalized


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        normalized_alias = canonicalize_column(alias)
        if normalized_alias in df.columns:
            return normalized_alias
    return None


def read_any_csv(path: Path) -> pd.DataFrame:
    if path.is_dir():
        csv_files = sorted(path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {path}")
        return pd.read_csv(csv_files[0])

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if not csv_names:
                raise FileNotFoundError(f"No CSV files found inside {path}")
            with archive.open(csv_names[0]) as f:
                return pd.read_csv(f)

    return pd.read_csv(path)


def download_kaggle_dataset(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ierad.zip"
    extract_dir = output_dir / "ierad_raw"

    if extract_dir.exists() and not force:
        return extract_dir

    try:
        subprocess.run([sys.executable, "-m", "kaggle", "--version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Kaggle CLI is not installed. Install dependencies with "
            "`/home/priya_paul/.venv/bin/pip install -r requirements.txt`."
        )

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(output_dir),
    ]
    subprocess.run(command, check=True)

    zip_files = sorted(output_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"Kaggle download did not create a zip file in {output_dir}")
    archive_path = zip_files[0]

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def series_or_default(df: pd.DataFrame, target: str, default):
    source = find_column(df, ALIASES[target])
    if source:
        return df[source]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series([default] * len(df), index=df.index)


def clean_priority(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = np.select(
        [
            values.str.fullmatch(r"(critical|life_threatening|p1|priority_1|1)", na=False),
            values.str.fullmatch(r"(high|urgent|major|p2|priority_2|2)", na=False),
            values.str.fullmatch(r"(low|minor|p4|priority_4|4)", na=False),
        ],
        ["Critical", "High", "Low"],
        default="Medium",
    )
    return pd.Series(mapped, index=series.index)


def clean_traffic(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = np.select(
        [
            values.str.contains("high|heavy|severe|congest", regex=True),
            values.str.contains("low|light|clear", regex=True),
        ],
        ["High", "Low"],
        default="Medium",
    )
    return pd.Series(mapped, index=series.index)


def clean_weather(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = np.select(
        [
            values.str.contains("snow|ice|sleet", regex=True),
            values.str.contains("fog|mist", regex=True),
            values.str.contains("rain|storm|drizzle", regex=True),
        ],
        ["Snowy", "Foggy", "Rainy"],
        default="Clear",
    )
    return pd.Series(mapped, index=series.index)


def clean_time_of_day(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        hour = parsed.dt.hour.fillna(12)
    else:
        numeric_hour = pd.to_numeric(series, errors="coerce")
        hour = numeric_hour.fillna(12)

    mapped = np.select(
        [
            (hour >= 5) & (hour < 12),
            (hour >= 12) & (hour < 17),
            (hour >= 17) & (hour < 22),
        ],
        ["Morning", "Afternoon", "Evening"],
        default="Night",
    )
    return pd.Series(mapped, index=series.index)


def clean_day_of_week(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        is_weekend = parsed.dt.dayofweek >= 5
        return pd.Series(np.where(is_weekend, "Weekend", "Weekday"), index=series.index)

    values = series.astype(str).str.strip().str.lower()
    return pd.Series(np.where(values.str.contains("sat|sun|weekend", regex=True), "Weekend", "Weekday"), index=series.index)


def clean_zone(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    if values.nunique(dropna=True) <= 10:
        return values.replace({"": "Unknown Zone", "nan": "Unknown Zone"})
    codes = pd.factorize(values.fillna("Unknown Zone"))[0] % 5
    return pd.Series([f"Zone {chr(65 + code)}" for code in codes], index=series.index)


def numeric_or_default(df: pd.DataFrame, target: str, default: float) -> pd.Series:
    source = find_column(df, ALIASES[target])
    if not source:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    values = pd.to_numeric(df[source], errors="coerce")
    return values.fillna(values.median() if values.notna().any() else default).astype(float)


def percentile_rank(series: pd.Series, inverse: bool = False) -> pd.Series:
    ranks = series.rank(pct=True).fillna(0.5)
    if inverse:
        return 1.0 - ranks
    return ranks


def operational_risk_score(output: pd.DataFrame) -> pd.Series:
    return (
        1.40 * percentile_rank(output["distance_to_scene"])
        + 0.90 * output["priority_level"].eq("Critical").astype(float)
        + 0.55 * output["priority_level"].eq("High").astype(float)
        + 0.95 * output["traffic_density"].eq("High").astype(float)
        + 0.35 * output["traffic_density"].eq("Medium").astype(float)
        + 0.65 * output["weather_conditions"].isin(["Snowy", "Foggy", "Rainy"]).astype(float)
        + 0.55 * percentile_rank(output["caller_stress_score"])
        + 0.55 * percentile_rank(output["crew_experience_years"], inverse=True)
        + 0.40 * percentile_rank(output["ambulance_age_years"], inverse=True)
        + 0.35 * percentile_rank(output["temperature"], inverse=True)
        + 0.35 * output["dispatch_zone"].astype(str).str.lower().str.contains("rural").astype(float)
    )


def infer_delay_risk(df: pd.DataFrame, output: pd.DataFrame) -> pd.Series:
    target_col = find_column(df, ALIASES["delay_risk"])
    if target_col:
        raw = df[target_col]
        if pd.api.types.is_numeric_dtype(raw):
            values = pd.to_numeric(raw, errors="coerce").fillna(0)
            if set(values.dropna().unique()).issubset({0, 1}):
                return values.astype(int)
            return (values > values.median()).astype(int)
        labels = raw.astype(str).str.lower()
        return labels.str.contains("yes|true|delayed|high|1", regex=True).astype(int)

    score = operational_risk_score(output)
    return (score >= score.quantile(0.65)).astype(int)


def convert_ierad_to_project_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    call_time = series_or_default(df, "call_time", "")
    incident_type = series_or_default(df, "incident_type", "Unknown Emergency").astype(str).str.strip()
    dispatch_zone = clean_zone(series_or_default(df, "dispatch_zone", "Zone A"))

    output = pd.DataFrame(
        {
            "case_id": series_or_default(df, "case_id", "CASE").astype(str),
            "priority_level": clean_priority(series_or_default(df, "priority_level", "Medium")),
            "incident_type": incident_type.replace({"": "Unknown Emergency", "nan": "Unknown Emergency"}),
            "time_of_day": clean_time_of_day(series_or_default(df, "time_of_day", call_time)),
            "weather_conditions": clean_weather(series_or_default(df, "weather_conditions", "Clear")),
            "day_of_week": clean_day_of_week(series_or_default(df, "day_of_week", call_time)),
            "traffic_density": clean_traffic(series_or_default(df, "traffic_density", "Medium")),
            "dispatch_zone": dispatch_zone,
            "distance_to_scene": numeric_or_default(df, "distance_to_scene", 5.0),
            "crew_experience_years": numeric_or_default(df, "crew_experience_years", 5.0),
            "ambulance_age_years": numeric_or_default(df, "ambulance_age_years", 4.0),
            "temperature": numeric_or_default(df, "temperature", 70.0),
            "caller_stress_score": numeric_or_default(df, "caller_stress_score", 5.0).clip(1, 10),
        }
    )

    output["delay_risk"] = infer_delay_risk(df, output)
    output["historical_zone_delay_rate"] = output.groupby("dispatch_zone")["delay_risk"].transform("mean")
    output["historical_zone_delay_rate"] = output["historical_zone_delay_rate"].fillna(output["delay_risk"].mean()).clip(0.01, 0.99)
    output["case_id"] = np.where(output["case_id"].isin(["CASE", "", "nan"]), "CASE_" + output.index.astype(str), output["case_id"])

    output = output[PROJECT_COLUMNS]
    return output.dropna().reset_index(drop=True)


def prepare_training_data(input_path: str | None, output_path: str, download: bool = False, force: bool = False) -> pd.DataFrame:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if download:
        raw_location = download_kaggle_dataset(output.parent / "raw", force=force)
        raw_df = read_any_csv(raw_location)
    elif input_path:
        raw_df = read_any_csv(Path(input_path))
    else:
        raise ValueError("Provide input_path or set download=True.")

    converted = convert_ierad_to_project_schema(raw_df)
    converted.to_csv(output, index=False)
    return converted


def main():
    parser = argparse.ArgumentParser(description="Import the Kaggle IERAD dataset into the project schema.")
    parser.add_argument("--input", help="Path to a downloaded IERAD CSV, ZIP, or directory of CSV files.")
    parser.add_argument("--output", default="data/emergency_response_data.csv", help="Output CSV path.")
    parser.add_argument("--download", action="store_true", help="Download IERAD with the Kaggle CLI before importing.")
    parser.add_argument("--force", action="store_true", help="Force a fresh Kaggle download.")
    args = parser.parse_args()

    df = prepare_training_data(args.input, args.output, download=args.download, force=args.force)
    print(f"Wrote {len(df):,} normalized records to {args.output}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
