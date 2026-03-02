import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from common_metrics import evaluate_recognition


# =============================================================================
# PATHS / PARAMS (explicit)
# =============================================================================

SRC_PATH = r"C:\Users\USER\manuscript-ocr\src"
TRBA_MODEL_PATH = r"C:\Users\USER\Desktop\OCR_MODELS\trba_yenisei_prt\best_acc_weights.onnx"
TRBA_CONFIG_PATH = r"C:\Users\USER\Desktop\OCR_MODELS\trba_yenisei_prt\config.json"
OUTPUT_DIR = r"C:\Users\USER\EAST_TRBA_simple_exp\benchmark_recognition\benchmark_results_trba"

ARCHIVES020525_IMAGES = r"C:\shared\ocr_datasets\YeniseiGovReports-PRT\val\img"
ARCHIVES020525_GT = r"C:\shared\ocr_datasets\YeniseiGovReports-PRT\val\YeniseiGovReports-PRT_gt.csv"

SCHOOL_NOTEBOOKS_RU_IMAGES = r"C:\shared\data0205\data02065\school_notebooks_RU\test_images"
SCHOOL_NOTEBOOKS_RU_GT = r"C:\shared\data0205\data02065\school_notebooks_RU\test_recognition.csv"

IAM_IMAGES = r"C:\shared\data0205\data02065\IAM\test_images"
IAM_GT = r"C:\shared\data0205\data02065\IAM\test_recognition.csv"


DATASETS = [
    {
        "name": "Archives020525",
        "images_dir": ARCHIVES020525_IMAGES,
        "gt_csv": ARCHIVES020525_GT,
        "has_header": True,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
    },
]

BATCH_SIZE = 16


if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from manuscript.recognizers import TRBA


def load_ground_truth(ds_cfg: dict) -> Dict[str, str]:
    gt = {}
    with open(ds_cfg["gt_csv"], "r", encoding=ds_cfg["encoding"], newline="") as f:
        reader = csv.reader(f)
        if ds_cfg["has_header"]:
            next(reader, None)

        image_col = ds_cfg["image_column"]
        text_col = ds_cfg["text_column"]
        for row in reader:
            if len(row) <= max(image_col, text_col):
                continue
            image_name = row[image_col].strip()
            text = row[text_col]
            gt[image_name] = text

    return gt


def run_dataset(recognizer: TRBA, ds_cfg: dict) -> dict:
    print(f"\n### DATASET: {ds_cfg['name']}")

    images_dir = Path(ds_cfg["images_dir"])
    gt_csv = Path(ds_cfg["gt_csv"])

    if not images_dir.exists() or not gt_csv.exists():
        print("Skip: missing images_dir or gt_csv")
        return {}

    ground_truths = load_ground_truth(ds_cfg)
    if not ground_truths:
        print("Skip: empty ground truth")
        return {}

    image_names: List[str] = []
    image_paths: List[str] = []

    for image_name in ground_truths.keys():
        image_path = images_dir / image_name
        if image_path.exists():
            image_names.append(image_name)
            image_paths.append(str(image_path))

    if not image_paths:
        print("Skip: no matching image files")
        return {}

    start = time.time()
    raw_predictions = recognizer.predict(images=image_paths, batch_size=BATCH_SIZE)
    elapsed = time.time() - start

    predictions = {}
    for image_name, prediction in zip(image_names, raw_predictions):
        predictions[image_name] = prediction.get("text", "")

    metrics = evaluate_recognition(
        predictions=predictions,
        ground_truths=ground_truths,
        lowercase=ds_cfg["lowercase"],
        normalize_unicode=ds_cfg["normalize_unicode"],
    )

    result = {
        "dataset": ds_cfg["name"],
        "model": "TRBA",
        "num_samples": len(metrics["matched_images"]),
        "num_gt": len(ground_truths),
        "num_predictions": len(predictions),
        "missing_predictions": metrics["missing_predictions"],
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "accuracy": metrics["accuracy"],
        "total_time_s": float(elapsed),
        "avg_time_ms": float((elapsed / len(image_paths)) * 1000.0),
        "throughput_fps": float(len(image_paths) / elapsed) if elapsed > 0 else 0.0,
    }

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    json_file = output_path / f"{ds_cfg['name']}_trba.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pred_csv = output_path / f"{ds_cfg['name']}_trba_predictions.csv"
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction"])
        for image_name in image_names:
            writer.writerow([image_name, predictions.get(image_name, "")])

    print(
        f"CER={result['cer']:.4f} | WER={result['wer']:.4f} | "
        f"ACC={result['accuracy']:.4f} | FPS={result['throughput_fps']:.2f}"
    )
    print(f"Saved: {json_file}")
    return result


def main() -> None:
    if not Path(TRBA_MODEL_PATH).exists():
        raise FileNotFoundError(f"TRBA model file not found: {TRBA_MODEL_PATH}")
    if not Path(TRBA_CONFIG_PATH).exists():
        raise FileNotFoundError(f"TRBA config file not found: {TRBA_CONFIG_PATH}")

    recognizer = TRBA(weights=TRBA_MODEL_PATH, config=TRBA_CONFIG_PATH)

    summary = []
    for ds in DATASETS:
        result = run_dataset(recognizer, ds)
        if result:
            summary.append(result)

    if summary:
        print("\n=== SUMMARY (TRBA) ===")
        for row in summary:
            print(
                f"{row['dataset']:<22} CER={row['cer']:.4f} "
                f"WER={row['wer']:.4f} ACC={row['accuracy']:.4f}"
            )
    else:
        print("No datasets were processed.")


if __name__ == "__main__":
    main()


