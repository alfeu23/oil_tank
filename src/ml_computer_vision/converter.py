import json
from pathlib import Path

with open("dataset/oil_tanks/fixed_coco/labels_coco_fixed.json") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

output_dir = Path("labels_yolo")
output_dir.mkdir(exist_ok=True)

for ann in data["annotations"]:
    image_id = ann["image_id"]
    bbox = ann["bbox"]

    img = images[image_id]

    w = img["width"]
    h = img["height"]

    x, y, bw, bh = bbox

    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    bw /= w
    bh /= h

    class_id = ann["category_id"] - 1

    txt_name = Path(img["file_name"]).stem + ".txt"

    with open(output_dir / txt_name, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")
