import csv
from pathlib import Path

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

image_path = Path(
    "Captura de Tela 2026-06-09 às 15.33.47.png",
)
output_dir = Path("output/images_captura")
output_dir.mkdir(parents=True, exist_ok=True)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="runs/detect/allfiles/weights/best.pt",
    confidence_threshold=0.9,
    device="mps",
    image_size=4800,
)

result = get_sliced_prediction(
    image_path.as_posix(),
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    postprocess_type="NMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.3,
)

result.export_visuals(
    output_dir.as_posix(),
    hide_labels=True,
    hide_conf=True,
)

img = Image.open(image_path).convert("RGB")
W, H = img.size

metadata = []

for i, pred in enumerate(result.object_prediction_list):
    bbox = pred.bbox

    x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])

    # margem para incluir sombra externa e contexto
    margin = 0.20
    bw = x2 - x1
    bh = y2 - y1

    x1m = max(0, int(x1 - bw * margin))
    y1m = max(0, int(y1 - bh * margin))
    x2m = min(W, int(x2 + bw * margin))
    y2m = min(H, int(y2 + bh * margin))

    crop = img.crop((x1m, y1m, x2m, y2m))
    crop = crop.resize((512, 512))

    crop_name = f"tank_{i:05d}.png"
    crop_path = output_dir / "crops" / crop_name
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)

    metadata.append(
        {
            "id": f"tank_{i:05d}",
            "source_image": str(image_path),
            "x1": x1m,
            "y1": y1m,
            "x2": x2m,
            "y2": y2m,
            "confidence": float(pred.score.value),
            "crop_path": str(crop_path),
        }
    )

with open("output/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
    writer.writeheader()
    writer.writerows(metadata)
