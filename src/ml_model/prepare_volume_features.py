import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME_CANDIDATES = [
    "Filename",
    "filename",
    "file_name",
    "image",
    "image_name",
    "source_filename",
]
VOLUME_CANDIDATES = [
    "volume_ratio",
    "volume_percent",
    "volume_percentage",
    "fill_ratio",
    "fill_percent",
    "mean_volume_ratio",
    "mean_volume_percent",
]


def _normalize_large_id(filename):
    value = Path(str(filename)).name
    value = value.replace("_large.jpg", "").replace("_large", "")
    return value.zfill(2) if value.isdigit() and int(value) < 100 else value


def _find_column(df, explicit, candidates, kind):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{kind} column not found: {explicit}")
        return explicit

    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not infer {kind} column. Available columns: "
        + ", ".join(map(str, df.columns))
    )


def _volume_to_ratio(series, column_name):
    values = pd.to_numeric(series, errors="coerce")
    column_lower = column_name.lower()
    if "percent" in column_lower or "percentage" in column_lower:
        values = values / 100.0
    elif values.dropna().gt(1.0).any():
        values = values / 100.0
    return values.clip(0.0, 1.0)


def prepare_volume_features(input_path, filename_column=None, volume_column=None):
    raw = pd.read_csv(input_path)
    filename_column = _find_column(raw, filename_column, FILENAME_CANDIDATES, "filename")
    volume_column = _find_column(raw, volume_column, VOLUME_CANDIDATES, "volume")

    df = raw.copy()
    df["Filename"] = df[filename_column].map(lambda value: Path(str(value)).name)
    df["large_id"] = df["Filename"].map(_normalize_large_id)
    df["volume_ratio"] = _volume_to_ratio(df[volume_column], volume_column)
    df = df.dropna(subset=["Filename", "volume_ratio"]).copy()
    if df.empty:
        raise ValueError("No valid image volume rows found after parsing the input CSV.")

    rows = []
    for filename, group in df.groupby("Filename", sort=True):
        valid = group["volume_ratio"].dropna()
        if valid.empty:
            continue
        row = {
            "Filename": filename,
            "large_id": _normalize_large_id(filename),
            "tank_count": int(len(group)),
            "valid_tank_count": int(len(valid)),
            "mean_fill_ratio": float(valid.mean()),
            "median_fill_ratio": float(valid.median()),
            "min_fill_ratio": float(valid.min()),
            "max_fill_ratio": float(valid.max()),
            "std_fill_ratio": float(valid.std(ddof=0)),
            "sum_outer_area_proxy": float(len(valid)),
            "storage_volume_proxy": float(valid.sum()),
            "empty_volume_proxy": float((1.0 - valid).sum()),
        }

        if "confidence" in group.columns:
            confidence = pd.to_numeric(group["confidence"], errors="coerce")
            row["mean_detection_confidence"] = (
                float(confidence.mean()) if confidence.notna().any() else np.nan
            )
        rows.append(row)

    if not rows:
        raise ValueError("No image-level volume features could be built from the input CSV.")
    return pd.DataFrame(rows).sort_values("Filename").reset_index(drop=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize image-level volume percentages into ML-Finance features."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV with one or more rows per large image and a volume percentage/ratio.",
    )
    parser.add_argument(
        "--output",
        default="financial_data/volume_features.csv",
        help="Output image-level feature CSV.",
    )
    parser.add_argument(
        "--filename-column",
        default=None,
        help="Optional explicit image filename column.",
    )
    parser.add_argument(
        "--volume-column",
        default=None,
        help="Optional explicit volume percentage/ratio column.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    features = prepare_volume_features(
        input_path=args.input,
        filename_column=args.filename_column,
        volume_column=args.volume_column,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} image-level feature rows to {output_path}")


if __name__ == "__main__":
    main()
