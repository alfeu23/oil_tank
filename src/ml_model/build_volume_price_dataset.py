import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


LARGE_VOLUME_RE = re.compile(r"large_(?P<large_id>\d+).*_volumes\.csv$")


def _normalize_large_id(raw_id):
    raw_id = str(raw_id).replace("_large.jpg", "").replace("_large", "")
    return raw_id.zfill(2) if raw_id.isdigit() and int(raw_id) < 100 else raw_id


def _image_filename_from_large_id(large_id):
    return f"{_normalize_large_id(large_id)}_large.jpg"


def _bool_series(series):
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _read_volume_file(path):
    match = LARGE_VOLUME_RE.search(path.name)
    if not match:
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    for column in [
        "external_area",
        "internal_shadow_area",
        "internal_shadow_ratio",
        "oil_area",
        "oil_ratio",
        "volume_ratio",
        "volume_percent",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "included_in_volume" in df.columns:
        included = _bool_series(df["included_in_volume"])
    else:
        included = df["volume_ratio"].notna()
    measured = df[included & df["volume_ratio"].notna()].copy()

    large_id = _normalize_large_id(match.group("large_id"))
    row = {
        "Filename": _image_filename_from_large_id(large_id),
        "large_id": large_id,
        "volume_csv": str(path),
        "tank_count": int(len(df)),
        "included_tank_count": int(len(measured)),
    }

    if measured.empty:
        row.update(
            {
                "mean_volume_ratio": np.nan,
                "median_volume_ratio": np.nan,
                "min_volume_ratio": np.nan,
                "max_volume_ratio": np.nan,
                "sum_external_area": 0.0,
                "storage_volume_proxy": 0.0,
                "empty_volume_proxy": 0.0,
                "mean_internal_shadow_ratio": np.nan,
                "shadowed_tank_count": 0,
            }
        )
        return row

    external_area = measured["external_area"].fillna(0.0)
    volume_ratio = measured["volume_ratio"].clip(0.0, 1.0).fillna(0.0)
    internal_shadow_ratio = measured["internal_shadow_ratio"].fillna(0.0)
    if "has_internal_shadow" in measured.columns:
        shadowed_tanks = int(_bool_series(measured["has_internal_shadow"]).sum())
    else:
        shadowed_tanks = int((internal_shadow_ratio > 0).sum())

    row.update(
        {
            "mean_volume_ratio": float(volume_ratio.mean()),
            "median_volume_ratio": float(volume_ratio.median()),
            "min_volume_ratio": float(volume_ratio.min()),
            "max_volume_ratio": float(volume_ratio.max()),
            "sum_external_area": float(external_area.sum()),
            "storage_volume_proxy": float((external_area * volume_ratio).sum()),
            "empty_volume_proxy": float((external_area * (1.0 - volume_ratio)).sum()),
            "mean_internal_shadow_ratio": float(internal_shadow_ratio.mean()),
            "shadowed_tank_count": shadowed_tanks,
        }
    )
    return row


def load_volume_features(volume_dir, volume_pattern):
    volume_paths = sorted(Path(volume_dir).glob(volume_pattern))
    rows = []
    for volume_path in volume_paths:
        row = _read_volume_file(volume_path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No large-image volume CSVs matched {Path(volume_dir) / volume_pattern}"
        )
    return pd.DataFrame(rows)


def load_precomputed_volume_features(volume_features_path):
    features = pd.read_csv(volume_features_path)
    if "Filename" not in features.columns:
        if "source_filename" in features.columns:
            features = features.rename(columns={"source_filename": "Filename"})
        else:
            raise ValueError(
                f"{volume_features_path} must contain Filename or source_filename"
            )

    if "large_id" not in features.columns:
        features["large_id"] = features["Filename"].map(_normalize_large_id)
    else:
        features["large_id"] = features["large_id"].map(_normalize_large_id)

    return features


def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path)
    if "Data" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a Data column with image dates")
    if "Filename" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a Filename column")

    metadata["image_date"] = pd.to_datetime(metadata["Data"], errors="coerce")
    metadata = metadata.dropna(subset=["image_date"]).copy()
    metadata["large_id"] = metadata["Filename"].map(_normalize_large_id)
    return metadata


def load_prices(prices_path, benchmark):
    prices = pd.read_csv(prices_path)
    if "benchmark" in prices.columns:
        prices = prices[prices["benchmark"].astype(str).str.lower() == benchmark]
    if prices.empty:
        raise ValueError(f"No {benchmark} price rows found in {prices_path}")

    prices["price_date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["oil_price_usd"] = pd.to_numeric(prices["oil_price_usd"], errors="coerce")
    prices = prices.dropna(subset=["price_date", "oil_price_usd"]).copy()
    return prices.sort_values("price_date")


def join_prices(dataset, prices, price_join):
    dataset = dataset.sort_values("image_date").copy()
    if price_join == "exact":
        joined = dataset.merge(
            prices[["price_date", "oil_price_usd"]],
            left_on="image_date",
            right_on="price_date",
            how="left",
        )
        missing_prices = joined["oil_price_usd"].isna()
        if missing_prices.any():
            missing_dates = sorted(
                joined.loc[missing_prices, "image_date"].dt.strftime("%Y-%m-%d").unique()
            )
            raise ValueError(
                "Some image dates do not exist in the price file: "
                + ", ".join(missing_dates)
            )
    else:
        direction = "nearest" if price_join == "nearest" else "backward"
        joined = pd.merge_asof(
            dataset,
            prices[["price_date", "oil_price_usd"]],
            left_on="image_date",
            right_on="price_date",
            direction=direction,
            tolerance=pd.Timedelta(days=7),
        )

    joined["price_date_lag_days"] = (
        joined["image_date"] - joined["price_date"]
    ).dt.days
    return joined.dropna(subset=["oil_price_usd"]).copy()


def build_dataset(
    metadata_path,
    prices_path,
    volume_features_path,
    volume_dir,
    volume_pattern,
    benchmark,
    price_join,
    include_missing_volumes,
):
    metadata = load_metadata(metadata_path)
    prices = load_prices(prices_path, benchmark)
    if volume_features_path:
        volume_features = load_precomputed_volume_features(volume_features_path)
    else:
        volume_features = load_volume_features(volume_dir, volume_pattern)

    how = "left" if include_missing_volumes else "inner"
    dataset = metadata.merge(volume_features, on=["Filename", "large_id"], how=how)
    dataset = join_prices(dataset, prices, price_join)
    dataset["day_of_year"] = dataset["image_date"].dt.dayofyear
    dataset["benchmark"] = benchmark

    first_columns = [
        "Filename",
        "large_id",
        "image_date",
        "price_date",
        "price_date_lag_days",
        "benchmark",
        "oil_price_usd",
    ]
    remaining_columns = [column for column in dataset.columns if column not in first_columns]
    dataset = dataset[first_columns + remaining_columns]
    return dataset.sort_values(["image_date", "Filename"]).reset_index(drop=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Join large-image tank volume estimates with image dates and oil prices."
    )
    parser.add_argument(
        "--metadata",
        default="large_image_data_with_dates.csv",
        help="Large image metadata CSV with Filename and Data columns.",
    )
    parser.add_argument(
        "--prices",
        default="financial_data/oil_prices_2018.csv",
        help="Oil price CSV produced by src/ml_model/download_oil_prices.py.",
    )
    parser.add_argument(
        "--volume-features",
        default="financial_data/volume_features.csv",
        help=(
            "Pre-aggregated image-level volume features, e.g. "
            "financial_data/volume_features.csv."
        ),
    )
    parser.add_argument(
        "--volume-dir",
        default="predictions",
        help="Directory containing classic_vision_large_*_volumes.csv files.",
    )
    parser.add_argument(
        "--volume-pattern",
        default="classic_vision_large_*_open_roof_volumes.csv",
        help="Glob for large-image volume CSVs.",
    )
    parser.add_argument("--benchmark", default="wti", help="Benchmark to keep from prices.")
    parser.add_argument(
        "--price-join",
        choices=["previous", "nearest", "exact"],
        default="exact",
        help="How to attach market prices to image dates.",
    )
    parser.add_argument(
        "--include-missing-volumes",
        action="store_true",
        help="Keep metadata rows even if a volume CSV has not been generated yet.",
    )
    parser.add_argument(
        "--output",
        default="financial_data/volume_price_dataset.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset = build_dataset(
        metadata_path=args.metadata,
        prices_path=args.prices,
        volume_features_path=args.volume_features,
        volume_dir=args.volume_dir,
        volume_pattern=args.volume_pattern,
        benchmark=args.benchmark.lower(),
        price_join=args.price_join,
        include_missing_volumes=args.include_missing_volumes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved {len(dataset)} joined rows to {output_path}")


if __name__ == "__main__":
    main()
