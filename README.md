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

After detecting circular tanks, the classic pipeline can estimate fill volume
from internal floating-roof shadow area:

```bash
.venv/bin/python src/classic_vision/main.py \
  --image "/Users/alfeu/.cache/kagglehub/datasets/towardsentropy/oil-storage-tanks/versions/1/Oil Tanks/image_patches/01_5_2.jpg" \
  --output predictions/classic_vision_volume.png \
  --calculate-volume \
  --min-radius 6 \
  --max-radius 40 \
  --min-area 40 \
  --max-area 6000
```

The volume estimate is:

```text
volume = 1 - (internal_shadow_area / external_tank_area)
```

Only tanks with open-roof evidence are included in the volume calculation by
default. A tank is treated as open roof when it has measurable internal shadow
or visible dark oil/liquid evidence inside the tank. Bright, uniform tanks
without either signal are marked as `closed_roof` so fixed/closed-roof tanks do
not skew the result. Use `--min-oil-fraction` to tune the oil evidence threshold
or `--include-unshadowed` only if you intentionally want unmeasured tanks counted
as 100% full.

Outputs:
- Annotated image with tank IDs and volume percentages.
- `*_volumes.csv` with per-tank area and volume values.
- `*_open_roof_evidence_mask.png` with the detected shadow/oil evidence pixels.
