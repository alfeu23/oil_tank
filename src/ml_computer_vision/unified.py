import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision.transforms.functional import to_tensor


def run_detection(
    image_path: Path,
    output_dir: Path,
    yolo_weights: Path,
    confidence_threshold: float = 0.9,
    device: str = "mps",
    detector_image_size: int = 4800,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_ratio: float = 0.1,
    crop_margin: float = 0.20,
    crop_size: int = 512,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=yolo_weights.as_posix(),
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=detector_image_size,
    )

    result = get_sliced_prediction(
        image_path.as_posix(),
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
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

        bw = x2 - x1
        bh = y2 - y1

        x1m = max(0, int(x1 - bw * crop_margin))
        y1m = max(0, int(y1 - bh * crop_margin))
        x2m = min(W, int(x2 + bw * crop_margin))
        y2m = min(H, int(y2 + bh * crop_margin))

        crop = img.crop((x1m, y1m, x2m, y2m))
        crop = crop.resize((crop_size, crop_size))

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

    return metadata


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool1(c1)

        c2 = self.down2(x)
        x = self.pool2(c2)

        x = self.bridge(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv1(x)

        return self.out(x)


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    0 = background   -> black
    1 = InnerShadow  -> green
    2 = OuterShadow  -> blue
    """
    colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    return colors[mask]


def overlay_mask(
    image: Image.Image, mask: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    img = np.array(image.convert("RGB"), dtype=np.float32)
    color_mask = mask_to_rgb(mask).astype(np.float32)

    active = mask > 0
    out = img.copy()
    out[active] = (1.0 - alpha) * img[active] + alpha * color_mask[active]
    return Image.fromarray(out.astype(np.uint8))


def segment_one(
    model: nn.Module,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = 512,
) -> dict:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size))

    x = to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay_mask(resized, pred).save(overlay_path)

    inner_area = int((pred == 1).sum())
    outer_area = int((pred == 2).sum())
    ratio = None if outer_area == 0 else 1.0 - (inner_area / outer_area)

    return {
        "overlay_path": str(overlay_path),
        "fill_ratio": ratio,
    }


def run_segmentation(
    crops_metadata: list[dict],
    unet_weights: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = 512,
) -> list[dict]:
    model = UNet(num_classes=3).to(device)
    state = torch.load(unet_weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    overlays_dir = output_dir / "overlays"

    ratios = []
    for entry in crops_metadata:
        crop_path = Path(entry["crop_path"])
        seg_result = segment_one(
            model, crop_path, overlays_dir, device, image_size=image_size
        )
        entry.update(seg_result)
        if seg_result["fill_ratio"] is not None:
            ratios.append(seg_result["fill_ratio"])

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
    else:
        avg_ratio = None

    return crops_metadata, avg_ratio


def write_metadata_csv(metadata: list[dict], csv_path: Path) -> None:
    if not metadata:
        print("Nenhum tanque detectado, CSV não gerado.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Metadata final salvo em: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline unificado: detecção de tanques (SAHI/YOLOv8) + segmentação de sombra (U-Net)."
    )
    parser.add_argument(
        "--image",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("runs/detect/allfiles/weights/best.pt"),
    )
    parser.add_argument(
        "--unet-weights", type=Path, default=Path("runs/unet/unet_tank_shadow.pt")
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--detector-device", type=str, default="mps")
    parser.add_argument(
        "--unet-device",
        type=str,
        default=None,
        help="cpu, mps, cuda, cuda:0. Default: auto.",
    )
    parser.add_argument("--detector-image-size", type=int, default=4800)
    parser.add_argument("--unet-image-size", type=int, default=512)
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Pula a etapa 1, assume que já existe metadata de crops.",
    )
    args = parser.parse_args()

    # Etapa 1: detecção + recorte
    metadata = run_detection(
        image_path=args.image,
        output_dir=args.output_dir,
        yolo_weights=args.yolo_weights,
        confidence_threshold=args.confidence_threshold,
        device=args.detector_device,
        detector_image_size=args.detector_image_size,
    )
    print(f"{len(metadata)} tanques detectados.")

    # Etapa 2: segmentação U-Net em cima dos crops
    unet_device = get_device(args.unet_device)
    print("device (U-Net):", unet_device)
    metadata, ratio = run_segmentation(
        crops_metadata=metadata,
        unet_weights=args.unet_weights,
        output_dir=args.output_dir,
        device=unet_device,
        image_size=args.unet_image_size,
    )

    # Etapa 3: CSV final unificado
    write_metadata_csv(metadata, args.output_dir / "metadata_full.csv")
    if ratio:
        write_metadata_csv(
            [{"average_fill_ratio": ratio}], args.output_dir / "average_fill_ratio.csv"
        )


if __name__ == "__main__":
    main()
