from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="runs/detect/allfiles/weights/best.pt",
    confidence_threshold=0.5,
    device="mps",
    image_size=4800,
)

result = get_sliced_prediction(
    "dataset/oil_tanks/large_images/01_large.jpg",
    detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    postprocess_type="NMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.3,
)

result.export_visuals(
    export_dir="predictions",
    hide_labels=True,
    hide_conf=True,
)
