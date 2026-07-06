**Goal**: Get a fully functioning Neural Network capable of getting images by google earth and categorizing the fullness of oil storage tanks by its shadow.
**Status**: Documenting and doing the Slide for Presentation.

## Tasks
- [x] Documenting and doing the Slide for Presentation.
- [x] Implement the first basic shadow calculator.
- [ ] Train the neural network to be able to on any image single-out the oil tanks and do the calculations on it.
- [ ] Bonus-Step: Make a integration with the united states oil regulation to try and predict the prices or information it will register.

## Technical Notes
-  Utilize the [Kaggle Dataset](https://www.kaggle.com/datasets/towardsentropy/oil-storage-tanks)
- "Automatic Oil Reserves Analysis Through the Shadows of Exterior Floating Crest Oil Tanks" - [Paper](https://www.researchgate.net/publication/311531636)
- Utilize [Google Earth Engine](https://earthengine.google.com/) -> As its the source of the images in the kaggle dataset.
- Utilize [Naraspace](https://ep.naraspace.com/post/contents/how-to-easily-grasp-global-oil-storage-volume) guide when felling lost.

## YOLOv8 (detecção) - preparação do dataset

O dataset original vem em formato COCO. Para treinar com YOLOv8 (Ultralytics) é mais prático ter a estrutura:

- `dataset/yolo/images/train/*.jpg`
- `dataset/yolo/images/val/*.jpg`
- `dataset/yolo/labels/train/*.txt`
- `dataset/yolo/labels/val/*.txt`

Além disso, é importante fazer o split **por imagem grande de origem** (prefixo `01_`, `02_`, ...), para evitar *data leakage* entre treino e validação.

### Gerar split train/val

```bash
python src/yolo/split_dataset.py --val-ratio 0.1 --seed 42
```

Saídas:
- `dataset/yolo/oil_tanks.yaml` (arquivo `data` para Ultralytics)
- `dataset/yolo/splits.json` (quais grupos foram para train/val)

## Classic vision volume estimate

The classic pipeline is configured for the project defaults:

```bash
.venv/bin/python src/classic_vision/main.py --patch 01_5_2
```

```bash
.venv/bin/python src/classic_vision/main.py --large 01
```

Modes:
- `--patch PATCH_ID`: runs `image_patches/PATCH_ID.jpg`, default `01_5_2`.
- `--large LARGE_ID`: runs `large_images/LARGE_ID_large.jpg`, default `01`.

For patch images, the pipeline uses the dataset's `Floating Head Tank` labels
when present. This identifies open-roof tanks directly, matching the Kaggle
notebook approach. If a patch has no `Floating Head Tank` labels, it falls back
to automatic circle detection plus open-roof filtering.

Volume estimation follows the Kaggle notebook's shadow-analysis method:
- Crop around the tank.
- Enhance shadows with `-(LAB_L + LAB_B) / (HSV_V + 1)`.
- Threshold with `0.6 * minimum_threshold + 0.4 * mean_threshold`.
- Clean the mask with border clearing, closing, hole filling, and connected
  component filtering.
- Select the two largest tank-shadow regions.

The volume estimate is:

```text
volume = 1 - (smaller_shadow_area / larger_shadow_area)
```

Outputs:
- Annotated image with tank IDs and volume percentages.
- `*_volumes.csv` with per-tank area and volume values.
- `*_open_roof_evidence_mask.png` with the detected shadow/oil evidence pixels.

## ML-Finance volume + price modeling

The ML-Finance pipeline starts from a machine-readable large-image volume CSV,
then joins it to image dates and oil prices.

Expected input columns:

- `Filename`: large image filename, e.g. `01_large.jpg`.
- one volume column such as `volume_percent`, `volume_percentage`,
  `volume_ratio`, `fill_percent`, or `fill_ratio`.

### Prepare ML volume features

```bash
uv run python src/ml_model/prepare_volume_features.py \
  --input path/to/large_image_volumes.csv \
  --output financial_data/volume_features.csv
```

Validate that dates and image-level features are usable for training:

```bash
uv run python src/ml_model/validate_training_data.py \
  --metadata large_image_data_with_dates.csv \
  --prices financial_data/oil_prices_2018.csv \
  --volume-features financial_data/volume_features.csv
```

### Download 2018 crude oil prices

Download daily WTI prices for 2018:

```bash
uv run python src/ml_model/download_oil_prices.py \
  --year 2018 \
  --benchmark wti \
  --output financial_data/oil_prices_2018.csv
```

Use `--benchmark brent` for Brent prices, or `--benchmark both` to keep both
series in the same file.

### Join image dates, volumes, and prices

```bash
uv run python src/ml_model/build_volume_price_dataset.py \
  --metadata large_image_data_with_dates.csv \
  --prices financial_data/oil_prices_2018.csv \
  --volume-features financial_data/volume_features.csv \
  --output financial_data/volume_price_dataset.csv
```

For image dates that fall on weekends or market holidays, the join uses the
exact trading-day date in `large_image_data_with_dates.csv`. If any image date
is missing from the price file, the command fails instead of silently dropping
rows.

### Train a price model

```bash
uv run python src/ml_model/train_price_model.py \
  --dataset financial_data/volume_price_dataset.csv \
  --output-dir financial_data/model
```

This trains a small ridge regression model from the image-level volume features
and date feature (`day_of_year`) to `oil_price_usd`. It also compares
against a mean-price baseline and a date-only baseline. It writes:

- `financial_data/model/model.json`
- `financial_data/model/metrics.csv`
- `financial_data/model/predictions.csv`

For a meaningful validation metric, prepare volume features for many large
images before training.
