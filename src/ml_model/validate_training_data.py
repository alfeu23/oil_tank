import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_VOLUME_COLUMNS = [
    "Filename",
    "tank_count",
    "valid_tank_count",
    "storage_volume_proxy",
    "empty_volume_proxy",
    "sum_outer_area_proxy",
]


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate_dates(metadata_path, prices_path):
    metadata = _load_csv(metadata_path)
    prices = _load_csv(prices_path)
    image_dates = set(metadata["Data"].astype(str))
    price_dates = set(prices["date"].astype(str))
    missing = sorted(image_dates - price_dates)
    if missing:
        raise ValueError(
            "Image dates missing from price file: " + ", ".join(missing)
        )
    print(f"Date check passed: {len(image_dates)} image dates are trading days.")


def validate_volume_features(volume_features_path):
    features = _load_csv(volume_features_path)
    missing_columns = [
        column for column in REQUIRED_VOLUME_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            "Volume feature file is missing columns: " + ", ".join(missing_columns)
        )

    duplicate_filenames = features["Filename"][features["Filename"].duplicated()].unique()
    if len(duplicate_filenames):
        raise ValueError(
            "Volume feature file has duplicate Filename rows: "
            + ", ".join(map(str, duplicate_filenames))
        )

    for column in [
        "storage_volume_proxy",
        "empty_volume_proxy",
        "sum_outer_area_proxy",
    ]:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    total_proxy = features["storage_volume_proxy"] + features["empty_volume_proxy"]
    consistency_error = np.abs(total_proxy - features["sum_outer_area_proxy"])
    tolerance = np.maximum(1.0, features["sum_outer_area_proxy"] * 0.01)
    inconsistent = features.loc[consistency_error > tolerance, "Filename"].tolist()
    if inconsistent:
        raise ValueError(
            "Storage and empty proxies do not match outer area proxy for: "
            + ", ".join(map(str, inconsistent[:10]))
        )

    print(f"Volume feature check passed: {len(features)} image-level rows.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate image dates and image-level ML features before training."
    )
    parser.add_argument("--metadata", default="large_image_data_with_dates.csv")
    parser.add_argument("--prices", default="financial_data/oil_prices_2018.csv")
    parser.add_argument(
        "--volume-features",
        default="financial_data/volume_features.csv",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    validate_dates(args.metadata, args.prices)
    validate_volume_features(args.volume_features)


if __name__ == "__main__":
    main()
