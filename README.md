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
