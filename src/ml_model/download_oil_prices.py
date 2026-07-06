import argparse
from pathlib import Path

import pandas as pd


FRED_SERIES = {
    "wti": ("DCOILWTICO", "WTI Crude Oil"),
    "brent": ("DCOILBRENTEU", "Brent Crude Oil"),
}


def _download_fred_series(benchmark, year):
    series_id, label = FRED_SERIES[benchmark]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_column = df.columns[0]
    value_column = df.columns[1]

    df = df.rename(columns={date_column: "date", value_column: "oil_price_usd"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["oil_price_usd"] = pd.to_numeric(df["oil_price_usd"], errors="coerce")
    df = df.dropna(subset=["date", "oil_price_usd"])
    df = df[df["date"].dt.year == year].copy()
    df["benchmark"] = benchmark
    df["series_id"] = series_id
    df["source"] = "FRED"
    df["description"] = label
    return df[["date", "benchmark", "series_id", "source", "description", "oil_price_usd"]]


def download_oil_prices(year, benchmarks):
    frames = [_download_fred_series(benchmark, year) for benchmark in benchmarks]
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["benchmark", "date"]).reset_index(drop=True)
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    return prices


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical daily crude oil prices for the ML dataset."
    )
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument(
        "--benchmark",
        choices=["wti", "brent", "both"],
        default="wti",
        help="Oil price benchmark to download.",
    )
    parser.add_argument(
        "--output",
        default="financial_data/oil_prices_2018.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    benchmarks = list(FRED_SERIES) if args.benchmark == "both" else [args.benchmark]
    prices = download_oil_prices(args.year, benchmarks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    print(f"Saved {len(prices)} price rows to {output_path}")


if __name__ == "__main__":
    main()
