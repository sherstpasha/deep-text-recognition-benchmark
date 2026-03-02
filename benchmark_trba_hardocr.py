import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from common_metrics import evaluate_recognition
from hardocr import DocumentOCRPipeline


# =============================================================================
# PATHS / PARAMS (explicit)
# =============================================================================

HARDOCR_CONFIG_PATH = r"config.yaml"
HARDOCR_MODEL_PATH = r"best_accuracy.pth"
OUTPUT_DIR = r"C:\Users\USER\EAST_TRBA_simple_exp\benchmark_recognition\benchmark_results_hardocr"

OCR_DATASETS_ROOT = r"C:\shared\ocr_datasets"

ORIG_CYRILLIC_IMAGES = rf"{OCR_DATASETS_ROOT}\CyrillicHandwritingDataset\test"
ORIG_CYRILLIC_GT = rf"{OCR_DATASETS_ROOT}\CyrillicHandwritingDataset\test.csv"

DONKEY_NUM_PRINTED_IMAGES = (
    rf"{OCR_DATASETS_ROOT}\DonkeySmallOCR-Numbers-Printed-15random\val\img"
)
DONKEY_NUM_PRINTED_GT = (
    rf"{OCR_DATASETS_ROOT}\DonkeySmallOCR-Numbers-Printed-15random\val.csv"
)

SCHOOL_NOTEBOOKS_RU_IMAGES = rf"{OCR_DATASETS_ROOT}\school_notebooks_RU\val"
SCHOOL_NOTEBOOKS_RU_GT = rf"{OCR_DATASETS_ROOT}\school_notebooks_RU\val_converted.csv"

ORIG_CYRILLIC_EXAMPLE_IMAGES = rf"{OCR_DATASETS_ROOT}\orig_cyrillic_example\test"
ORIG_CYRILLIC_EXAMPLE_GT = rf"{OCR_DATASETS_ROOT}\orig_cyrillic_example\test.csv"

YENISEI_HWR_IMAGES = rf"{OCR_DATASETS_ROOT}\YeniseiGovReports-HWR\val\img"
YENISEI_HWR_GT = rf"{OCR_DATASETS_ROOT}\YeniseiGovReports-HWR\val\labels.csv"

YENISEI_PRT_IMAGES = rf"{OCR_DATASETS_ROOT}\YeniseiGovReports-PRT\val\img"
YENISEI_PRT_GT = rf"{OCR_DATASETS_ROOT}\YeniseiGovReports-PRT\val\labels.csv"

RUSSIAN_SCHOOL_ESSAYS_IMAGES = rf"{OCR_DATASETS_ROOT}\RussianSchoolEssays\test\test_images"
RUSSIAN_SCHOOL_ESSAYS_GT = rf"{OCR_DATASETS_ROOT}\RussianSchoolEssays\test\test.csv"

HANDWRITTEN_KAZAKH_RUSSIAN_IMAGES = rf"{OCR_DATASETS_ROOT}\HandwrittenKazakhRussian\val"
HANDWRITTEN_KAZAKH_RUSSIAN_GT = rf"{OCR_DATASETS_ROOT}\HandwrittenKazakhRussian\val\gt.txt"


DATASETS = [
    {
        "name": "orig_cyrillic",
        "images_dir": ORIG_CYRILLIC_IMAGES,
        "gt_csv": ORIG_CYRILLIC_GT,
        "has_header": False,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "DonkeySmallOCR-Numbers-Printed-15random",
        "images_dir": DONKEY_NUM_PRINTED_IMAGES,
        "gt_csv": DONKEY_NUM_PRINTED_GT,
        "has_header": True,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "school_notebooks_RU",
        "images_dir": SCHOOL_NOTEBOOKS_RU_IMAGES,
        "gt_csv": SCHOOL_NOTEBOOKS_RU_GT,
        "has_header": True,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "orig_cyrillic_example",
        "images_dir": ORIG_CYRILLIC_EXAMPLE_IMAGES,
        "gt_csv": ORIG_CYRILLIC_EXAMPLE_GT,
        "has_header": False,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "YeniseiGovReports-HWR",
        "images_dir": YENISEI_HWR_IMAGES,
        "gt_csv": YENISEI_HWR_GT,
        "has_header": True,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "YeniseiGovReports-PRT",
        "images_dir": YENISEI_PRT_IMAGES,
        "gt_csv": YENISEI_PRT_GT,
        "has_header": True,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "RussianSchoolEssays",
        "images_dir": RUSSIAN_SCHOOL_ESSAYS_IMAGES,
        "gt_csv": RUSSIAN_SCHOOL_ESSAYS_GT,
        "has_header": False,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
    {
        "name": "HandwrittenKazakhRussian",
        "images_dir": HANDWRITTEN_KAZAKH_RUSSIAN_IMAGES,
        "gt_csv": HANDWRITTEN_KAZAKH_RUSSIAN_GT,
        "has_header": False,
        "image_column": 0,
        "text_column": 1,
        "encoding": "utf-8",
        "lowercase": True,
        "normalize_unicode": "NFC",
        "delimiter": ",",
    },
]


def load_ground_truth(ds_cfg: dict) -> Dict[str, str]:
    gt = {}
    with open(ds_cfg["gt_csv"], "r", encoding=ds_cfg["encoding"], newline="") as f:
        reader = csv.reader(f, delimiter=ds_cfg.get("delimiter", ","))
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


def run_dataset(recognizer: DocumentOCRPipeline, ds_cfg: dict) -> dict:
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
    predictions = {}
    for image_name, image_path in zip(image_names, image_paths):
        try:
            with Image.open(image_path) as image:
                raw_text = recognizer._recognize_text(image)
            predictions[image_name] = recognizer._clean_text(raw_text)
        except Exception as exc:
            print(f"[WARN] Failed to recognize '{image_name}': {exc}")
    elapsed = time.time() - start

    metrics = evaluate_recognition(
        predictions=predictions,
        ground_truths=ground_truths,
        lowercase=ds_cfg["lowercase"],
        normalize_unicode=ds_cfg["normalize_unicode"],
    )

    result = {
        "dataset": ds_cfg["name"],
        "model": "DocumentOCRPipeline._recognize_text",
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

    json_file = output_path / f"{ds_cfg['name']}_hardocr.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pred_csv = output_path / f"{ds_cfg['name']}_hardocr_predictions.csv"
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
    if not Path(HARDOCR_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"HARDOCR model file not found: {Path(HARDOCR_MODEL_PATH).resolve()}"
        )
    if not Path(HARDOCR_CONFIG_PATH).exists():
        raise FileNotFoundError(
            f"HARDOCR config file not found: {Path(HARDOCR_CONFIG_PATH).resolve()}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = DocumentOCRPipeline(
        config_path=HARDOCR_CONFIG_PATH,
        ocr_model_path=HARDOCR_MODEL_PATH,
        device=device,
        batch_size=1,
    )

    summary = []
    for ds in DATASETS:
        result = run_dataset(recognizer, ds)
        if result:
            summary.append(result)

    if summary:
        print("\n=== SUMMARY (HARDOCR) ===")
        for row in summary:
            print(
                f"{row['dataset']:<22} CER={row['cer']:.4f} "
                f"WER={row['wer']:.4f} ACC={row['accuracy']:.4f}"
            )
    else:
        print("No datasets were processed.")


if __name__ == "__main__":
    main()
