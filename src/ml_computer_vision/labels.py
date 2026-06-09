import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

coco_path = Path("project-1-at-2026-06-09-13-40-beae498f/result.json")
coco_image_path = Path("project-1-at-2026-06-09-13-40-beae498f/images")
out_dir = Path("dataset_seg")
mask_dir = out_dir / "masks"
mask_dir.mkdir(parents=True, exist_ok=True)

with open(coco_path) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

# COCO category_id -> valor na máscara
# 0 no COCO vira 1 na máscara; 1 no COCO vira 2 na máscara
class_map = {
    0: 1,  # InnerShadow
    1: 2,  # OuterShadow
}

anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

for image_id, img_info in images.items():
    width = img_info["width"]
    height = img_info["height"]

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for ann in anns_by_image.get(image_id, []):
        class_value = class_map[ann["category_id"]]

        for polygon in ann["segmentation"]:
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=class_value)

    original_name = Path(img_info["file_name"]).name
    # remove prefixo aleatório do Label Studio, mantendo tank_00000.png
    if "-" in original_name:
        original_name = original_name.split("-", 1)[1]

    mask_name = Path(original_name).with_suffix(".png").name
    mask.save(mask_dir / mask_name)

print("Máscaras criadas em:", mask_dir)


mask = np.array(Image.open("dataset_seg/masks/tank_00000.png"))
print(np.unique(mask))

Path("dataset_seg/masks_vis").mkdir(parents=True, exist_ok=True)
vis = (mask * 120).astype(np.uint8)
Image.fromarray(vis).save("dataset_seg/masks_vis/tank_00000_vis.png")
