import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "storage_volume_proxy",
    "empty_volume_proxy",
    "mean_fill_ratio",
    "median_fill_ratio",
    "valid_tank_count",
    "sum_outer_area_proxy",
    "mean_detection_confidence",
    "std_fill_ratio",
    "tank_count",
    "day_of_year",
]
DATE_ONLY_FEATURES = ["day_of_year"]


def _prepare_features(df, requested_features):
    available = [feature for feature in requested_features if feature in df.columns]
    if not available:
        raise ValueError("None of the requested feature columns exist in the dataset")

    X = df[available].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians)
    return X.astype(float), available, medians


def _fit_ridge(X, y, alpha):
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds = np.where(stds == 0, 1.0, stds)
    X_scaled = (X - means) / stds
    X_augmented = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    penalty = np.eye(X_augmented.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        X_augmented.T @ X_augmented + penalty,
        X_augmented.T @ y,
    )
    return coefficients, means, stds


def _predict(X, coefficients, means, stds):
    X_scaled = (X - means) / stds
    X_augmented = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    return X_augmented @ coefficients


def _metrics(y_true, y_pred):
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual = float(np.sum(errors**2))
    r2 = float(1.0 - (residual / total)) if total > 0 else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _split_counts(row_count):
    if row_count < 2:
        raise ValueError(
            "Need at least 2 image-level rows to train. Prepare ML-Finance "
            "volume features for more large images first."
        )
    test_count = max(1, int(round(row_count * 0.2))) if row_count >= 5 else 0
    return row_count - test_count, test_count


def _evaluate_predictions(y, predictions, train_count, test_count):
    metrics = {"train": _metrics(y[:train_count], predictions[:train_count])}
    if test_count:
        metrics["test"] = _metrics(y[train_count:], predictions[train_count:])
    else:
        metrics["test"] = None
    return metrics


def _train_ridge_named(df, y, train_count, test_count, target, features, alpha):
    X_df, feature_names, medians = _prepare_features(df, features)
    X = X_df.to_numpy(dtype=float)
    X_train = X[:train_count]
    y_train = y[:train_count]

    coefficients, means, stds = _fit_ridge(X_train, y_train, alpha)
    predictions = _predict(X, coefficients, means, stds)
    return {
        "algorithm": "ridge_regression",
        "alpha": alpha,
        "target": target,
        "features": feature_names,
        "feature_medians": medians.to_dict(),
        "feature_means": dict(zip(feature_names, means.tolist())),
        "feature_stds": dict(zip(feature_names, stds.tolist())),
        "intercept": float(coefficients[0]),
        "coefficients": dict(zip(feature_names, coefficients[1:].tolist())),
        "metrics": _evaluate_predictions(y, predictions, train_count, test_count),
        "predictions": predictions,
    }


def _mean_baseline(y, train_count, test_count):
    train_mean = float(np.mean(y[:train_count]))
    predictions = np.full(len(y), train_mean, dtype=float)
    return {
        "algorithm": "train_mean",
        "constant_prediction": train_mean,
        "metrics": _evaluate_predictions(y, predictions, train_count, test_count),
        "predictions": predictions,
    }


def _drop_predictions(model):
    clean_model = model.copy()
    clean_model.pop("predictions", None)
    return clean_model


def train_price_model(dataset_path, output_dir, target, features, alpha):
    df = pd.read_csv(dataset_path)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()
    if "image_date" in df.columns:
        df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
        df = df.sort_values(["image_date", "Filename"]).reset_index(drop=True)

    y = df[target].to_numpy(dtype=float)
    train_count, test_count = _split_counts(len(df))

    mean_model = _mean_baseline(y, train_count, test_count)
    project_model = _train_ridge_named(
        df,
        y,
        train_count,
        test_count,
        target,
        features,
        alpha,
    )

    if all(feature in df.columns for feature in DATE_ONLY_FEATURES):
        date_model = _train_ridge_named(
            df,
            y,
            train_count,
            test_count,
            target,
            DATE_ONLY_FEATURES,
            alpha,
        )
    else:
        date_model = None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions = df.copy()
    predictions["predicted_mean_price_usd"] = mean_model["predictions"]
    if date_model is not None:
        predictions["predicted_date_only_price_usd"] = date_model["predictions"]
    else:
        predictions["predicted_date_only_price_usd"] = np.nan
    predictions["predicted_oil_price_usd"] = project_model["predictions"]
    predictions["price_residual_usd"] = project_model["predictions"] - y
    predictions.to_csv(output_path / "predictions.csv", index=False)

    model = {
        "target": target,
        "row_count": int(len(df)),
        "train_count": int(train_count),
        "test_count": int(test_count),
        "split": "chronological",
        "models": {
            "mean_baseline": _drop_predictions(mean_model),
            "date_only": _drop_predictions(date_model) if date_model else None,
            "storage_feature_model": _drop_predictions(project_model),
        },
        "metrics": project_model["metrics"],
    }
    with open(output_path / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    metrics_rows = []
    for name, fitted_model in model["models"].items():
        if fitted_model is None:
            continue
        for split_name, split_metrics in fitted_model["metrics"].items():
            if split_metrics is None:
                continue
            metrics_rows.append({"model": name, "split": split_name, **split_metrics})
    pd.DataFrame(metrics_rows).to_csv(output_path / "metrics.csv", index=False)

    return model


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train oil price baselines and a storage-feature ML regression model."
    )
    parser.add_argument(
        "--dataset",
        default="financial_data/volume_price_dataset.csv",
        help="Joined dataset produced by build_volume_price_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="financial_data/model",
        help="Directory for model.json, metrics.csv, and predictions.csv.",
    )
    parser.add_argument("--target", default="oil_price_usd")
    parser.add_argument(
        "--features",
        nargs="*",
        default=DEFAULT_FEATURES,
        help="Numeric feature columns to use for the storage-feature model.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    model = train_price_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        target=args.target,
        features=args.features,
        alpha=args.alpha,
    )
    print(json.dumps(model["models"], indent=2))
    print(f"Saved model to {Path(args.output_dir) / 'model.json'}")


if __name__ == "__main__":
    main()
