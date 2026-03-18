"""
var1: визуализация боксов с моделью из ЭТОГО репозитория (CRAFT + TRBA).
Проходит по всем фото в папке, сохраняет {имя}_var_1.jpg
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from PIL import Image, ImageDraw
from recog_tools import load_model, detect_image_craft

# ── настройки ─────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
OCR_MODEL_PATH = "model_095.pth"
IMAGES_DIR = Path(r"C:\Users\USER\Desktop\images")
# ──────────────────────────────────────────────────────────────────


def visualize_boxes(pil_img, word_boxes, box_color=(0, 120, 255, 90),
                    outline_color=(0, 120, 255, 180), outline_width=2):
    """
    Рисует полупрозрачные синие боксы на изображении.
    word_boxes: list of (x1, y1, x2, y2)
    """
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (x1, y1, x2, y2) in word_boxes:
        draw.rectangle([x1, y1, x2, y2], fill=box_color,
                       outline=outline_color, width=outline_width)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def get_word_boxes(image_np):
    """
    CRAFT детекция → список (x1, y1, x2, y2) для каждого слова.
    """
    gpu = torch.cuda.is_available()
    final_boxes = detect_image_craft(image_np, gpu=gpu)

    word_rects = []
    for line in final_boxes:
        for poly in line:
            pts = np.array(poly, dtype=np.int32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            word_rects.append((int(x1), int(y1), int(x2), int(y2)))
    return word_rects


if __name__ == "__main__":
    model = load_model(CONFIG_PATH, OCR_MODEL_PATH)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(f for f in IMAGES_DIR.iterdir()
                   if f.suffix.lower() in exts and "_var_" not in f.stem)
    print(f"[INFO] Найдено {len(files)} изображений в {IMAGES_DIR}")

    for fpath in files:
        image_np = cv2.imread(str(fpath))
        if image_np is None:
            print(f"[SKIP] Не удалось прочитать {fpath.name}")
            continue

        start = time.perf_counter()
        word_boxes = get_word_boxes(image_np)
        elapsed = time.perf_counter() - start

        pil_img = Image.open(fpath)
        vis = visualize_boxes(pil_img, word_boxes)

        out_name = f"{fpath.stem}_var_1.jpg"
        out_path = IMAGES_DIR / out_name
        vis.save(str(out_path))

        print(f"[OK] {fpath.name} → {out_name}  ({elapsed:.2f}s, {len(word_boxes)} words)")
