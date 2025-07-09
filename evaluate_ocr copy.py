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
    (
        r"C:\data_19_04\Archive_19_04\data_archive\gt_test.txt",
        r"C:\data_19_04\Archive_19_04\data_archive",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_archive\gt_train.txt",
        r"C:\data_19_04\Archive_19_04\data_archive",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_cyrillic\gt_test.txt",
        r"C:\data_19_04\Archive_19_04\data_cyrillic\test",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_cyrillic\gt_train.txt",
        r"C:\data_19_04\Archive_19_04\data_cyrillic\train",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_hkr\gt_test.txt",
        r"C:\data_19_04\Archive_19_04\data_hkr\test",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_hkr\gt_train.txt",
        r"C:\data_19_04\Archive_19_04\data_hkr\train",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_school\gt_test.txt",
        r"C:\data_19_04\Archive_19_04\data_school",
    ),
    (
        r"C:\data_19_04\Archive_19_04\data_school\gt_train.txt",
        r"C:\data_19_04\Archive_19_04\data_school",
    ),
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
    gt_txt: str, img_dir: str, pipeline: DocumentOCRPipeline, batch_size: int = 512
) -> pd.DataFrame:
    results = []
    total = 0

    with open(gt_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and "\t" in l]

    samples = []
    for line in lines:
        img_name, gt_text = line.split("\t")
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            samples.append((img_name, gt_text, img_path))
        else:
            print(f"[WARN] Image not found: {img_path}")

    print(f"[INFO] Evaluating {gt_txt} ({len(samples)} samples)...")

    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch_samples = samples[i : i + batch_size]
        images = []
        meta = []

        for img_name, gt_text, img_path in batch_samples:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                meta.append((img_name, gt_text.strip()))
            except Exception as e:
                print(f"[ERROR] Failed to open {img_path}: {e}")

        preds = pipeline._recognize_batch_texts(images)

        for (img_name, gt), pred_text in zip(meta, preds):
            gt_lower = gt.lower()
            pred_lower = pred_text.strip().lower()

            norm_acc = compute_norm_accuracy(gt, pred_text)
            norm_acc_lower = compute_norm_accuracy(gt_lower, pred_lower)

            results.append(
                {
                    "index": total,
                    "image": img_name,
                    "gt": gt,
                    "pred": pred_text,
                    "correct": int(gt == pred_text),
                    "norm_accuracy": norm_acc,
                    "correct_lower": int(gt_lower == pred_lower),
                    "norm_accuracy_lower": norm_acc_lower,
                    "is_wrong": int(gt != pred_text),
                }
            )
            total += 1

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
        df["dataset"] = Path(gt_path).stem  # ← короткое имя датасета
        df["gt_file"] = gt_path  # ← полный путь GT
        all_results.append(df)

    if not all_results:
        print("[ERROR] No results collected.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # === Сводка по dataset (например: data_cyrillic, data_hkr)
    summary_by_dataset = (
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

    # === Сводка по gt-файлу (полный путь)
    summary_by_gtfile = (
        final_df.groupby("gt_file")[
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

    # === Общая финальная строка
    summary_total = pd.DataFrame(
        {
            "dataset": ["TOTAL"],
            "accuracy": [final_df["correct"].sum() / len(final_df)],
            "accuracy_lower": [final_df["correct_lower"].sum() / len(final_df)],
            "norm_accuracy": [final_df["norm_accuracy"].mean()],
            "norm_accuracy_lower": [final_df["norm_accuracy_lower"].mean()],
        }
    )

    # === Сохраняем всё в Excel
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel) as writer:
        final_df.to_excel(writer, sheet_name="details", index=False)
        summary_by_dataset.to_excel(writer, sheet_name="by_dataset", index=False)
        summary_by_gtfile.to_excel(writer, sheet_name="by_gt_file", index=False)
        summary_total.to_excel(writer, sheet_name="total", index=False)

        wrong_preds = final_df[final_df["is_wrong"] == 1]
        wrong_preds.to_excel(writer, sheet_name="wrong_preds", index=False)

    print(f"[DONE] Results saved to: {output_excel}")


if __name__ == "__main__":
    main()
