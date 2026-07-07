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
    image_stem = image_path.stem  # evita colisão de nomes entre imagens diferentes

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

        crop_name = f"{image_stem}_tank_{i:05d}.png"
        crop_path = output_dir / "crops" / image_stem / crop_name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)

        metadata.append(
            {
                "id": f"{image_stem}_tank_{i:05d}",
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


def compile_fill_ratios(metadata: list[dict]) -> list[dict]:
    """Compila, num único CSV, o fill_ratio de todos os tanques de todas as
    imagens (sem agrupar/tirar média por imagem)."""
    rows = []
    for entry in metadata:
        rows.append(
            {
                "id": entry.get("id"),
                "source_image": entry.get("source_image"),
                "fill_ratio": entry.get("fill_ratio"),
            }
        )
    return rows


def compute_average_by_image(metadata: list[dict]) -> list[dict]:
    """Agrupa o metadata por source_image e calcula a média do fill_ratio
    em cada imagem, ignorando tanques sem ratio válido (OuterShadow = 0)."""
    by_image: dict[str, list[float]] = {}
    counts: dict[str, int] = {}

    for entry in metadata:
        source = entry["source_image"]
        counts[source] = counts.get(source, 0) + 1
        ratio = entry.get("fill_ratio")
        if ratio is not None:
            by_image.setdefault(source, []).append(ratio)

    rows = []
    for source in sorted(counts.keys()):
        ratios = by_image.get(source, [])
        avg = sum(ratios) / len(ratios) if ratios else None
        rows.append(
            {
                "source_image": source,
                "num_tanks": counts[source],
                "num_valid_ratio": len(ratios),
                "average_fill_ratio": avg,
            }
        )

    return rows


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


def resolve_image_list(
    image_args: list[Path], images_dir: Optional[Path], pattern: str
) -> list[Path]:
    """Monta a lista final de imagens a processar, a partir de:
    - --image (um ou mais caminhos; o shell já expande glob antes de chegar aqui)
    - --images-dir + --pattern (glob feito pelo próprio script)
    """
    images: list[Path] = list(image_args)

    if images_dir is not None:
        images.extend(sorted(images_dir.glob(pattern)))

    seen = set()
    unique_images = []
    for p in images:
        if p not in seen:
            seen.add(p)
            unique_images.append(p)

    return unique_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline unificado: detecção de tanques (SAHI/YOLOv8) + segmentação de sombra (U-Net)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=None,
        help="Uma ou mais imagens. Um glob (ex: dataset/*.jpg) já é expandido pelo "
        "próprio shell em múltiplos arquivos.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Pasta com várias imagens a processar (alternativa a --image com glob).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Padrão glob usado junto com --images-dir (default: *.jpg).",
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
    args = parser.parse_args()

    image_list = resolve_image_list(args.image or [], args.images_dir, args.pattern)
    if not image_list:
        raise SystemExit("Nenhuma imagem informada: use --image ou --images-dir.")

    print(f"Processando {len(image_list)} imagem(ns):")
    for p in image_list:
        print(f"  - {p}")

    # Etapa 1: detecção + recorte, para cada imagem
    all_metadata: list[dict] = []
    for image_path in image_list:
        if not image_path.exists():
            print(f"[aviso] imagem não encontrada, pulando: {image_path}")
            continue
        print(f"-- detectando: {image_path}")
        metadata = run_detection(
            image_path=image_path,
            output_dir=args.output_dir,
            yolo_weights=args.yolo_weights,
            confidence_threshold=args.confidence_threshold,
            device=args.detector_device,
            detector_image_size=args.detector_image_size,
        )
        print(f"   {len(metadata)} tanques detectados em {image_path.name}.")
        all_metadata.extend(metadata)

    print(f"Total de tanques detectados em todas as imagens: {len(all_metadata)}")

    # Etapa 2: segmentação U-Net em cima de todos os crops de todas as imagens
    unet_device = get_device(args.unet_device)
    print("device (U-Net):", unet_device)
    all_metadata, ratio = run_segmentation(
        crops_metadata=all_metadata,
        unet_weights=args.unet_weights,
        output_dir=args.output_dir,
        device=unet_device,
        image_size=args.unet_image_size,
    )

    # Etapa 3: CSVs finais
    write_metadata_csv(all_metadata, args.output_dir / "metadata_full.csv")

    # CSV único com todos os fill_ratio de todos os tanques/imagens compilados
    fill_ratios_compiled = compile_fill_ratios(all_metadata)
    write_metadata_csv(
        fill_ratios_compiled, args.output_dir / "fill_ratios_compiled.csv"
    )

    # CSV com a média do fill_ratio compilada por imagem
    averages_by_image = compute_average_by_image(all_metadata)
    write_metadata_csv(
        averages_by_image, args.output_dir / "average_fill_ratio_by_image.csv"
    )

    if ratio is not None:
        print(f"Average fill ratio (todas as imagens): {ratio:.4f}")


if __name__ == "__main__":
    main()
