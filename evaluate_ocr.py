import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from Levenshtein import distance as levenshtein_distance
from hardocr import DocumentOCRPipeline
from tqdm import tqdm

# === ПАРАМЕТРЫ ===
config_path = r"config.yaml"
model_path = r"C:\Users\pasha\OneDrive\Рабочий стол\best_accuracy.pth"
datasets: List[Tuple[str, str]] = [  # (gt_path, img_folder)
    (r"C:\data_cyrillic\gt_test.txt", r"C:\data_cyrillic\test")
]
output_excel = r"C:\output\ocr_eval.xlsx"
# =================


def compute_norm_accuracy(gt: str, pred: str) -> float:
    """1 - нормализованное расстояние редактирования (чем ближе к 1 — тем лучше)"""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    ed = levenshtein_distance(gt, pred)
    return 1.0 - min(1.0, ed / max(len(gt), len(pred)))


def evaluate_dataset(
    gt_txt: str, img_dir: str, pipeline: DocumentOCRPipeline
) -> pd.DataFrame:
    results = []
    total = 0

    with open(gt_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"[INFO] Evaluating {gt_txt} ({len(lines)} samples)...")

    for line in tqdm(lines, desc="Processing crops", unit="img"):
        if "\t" not in line:
            continue
        img_name, gt_text = line.split("\t")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            word_resp = pipeline.recognize_word(
                img, box=(0, 0, w, h), word_idx=0, score=1.0
            )
            pred_text = word_resp.text

            gt = gt_text.strip()
            pred = pred_text.strip()

            gt_lower = gt.lower()
            pred_lower = pred.lower()

            norm_acc = compute_norm_accuracy(gt, pred)
            norm_acc_lower = compute_norm_accuracy(gt_lower, pred_lower)

            results.append(
                {
                    "index": total,
                    "image": img_name,
                    "gt": gt,
                    "pred": pred,
                    "correct": int(gt == pred),
                    "norm_accuracy": norm_acc,
                    "correct_lower": int(gt_lower == pred_lower),
                    "norm_accuracy_lower": norm_acc_lower,
                    "is_wrong": int(gt != pred),
                }
            )

            total += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {img_name}: {e}")
            continue

    return pd.DataFrame(results)


def main():
    pipeline = DocumentOCRPipeline(config_path, model_path, device="cuda")
    all_results = []

    for gt_path, img_path in datasets:
        if not os.path.exists(gt_path):
            print(f"[SKIP] GT file not found: {gt_path}")
            continue
        if not os.path.isdir(img_path):
            print(f"[SKIP] Image directory not found: {img_path}")
            continue

        df = evaluate_dataset(gt_path, img_path, pipeline)
        df["dataset"] = Path(gt_path).stem
        all_results.append(df)

    if not all_results:
        print("[ERROR] No results collected.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # === Сводная таблица
    summary_df = (
        final_df.groupby("dataset")[
            ["correct", "correct_lower", "norm_accuracy", "norm_accuracy_lower"]
        ]
        .agg(
            accuracy=("correct", lambda x: x.sum() / len(x)),
            accuracy_lower=("correct_lower", lambda x: x.sum() / len(x)),
            norm_accuracy=("norm_accuracy", "mean"),
            norm_accuracy_lower=("norm_accuracy_lower", "mean"),
        )
        .reset_index()
    )

    # === Сохраняем всё
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel) as writer:
        final_df.to_excel(writer, sheet_name="details", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        wrong_preds = final_df[final_df["is_wrong"] == 1]
        wrong_preds.to_excel(writer, sheet_name="wrong_preds", index=False)

    print(f"[DONE] Results saved to: {output_excel}")


if __name__ == "__main__":
    main()
