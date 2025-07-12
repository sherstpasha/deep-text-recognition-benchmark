import numpy as np
from PIL import Image
import torch
import time  # ← добавлено

from hardocr import DocumentOCRPipeline

# настройка устройства, модели и конфига
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "config.yaml"
ocr_model_path = r"C:\Users\pasha\OneDrive\Рабочий стол\best_accuracy.pth"
image_path = r"C:\output\1410.jpg"

# Создание пайплайна
pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
    TTA=True,
    TTA_thresh=0.1,
    use_nms=True,
    batch_size=8,
)

# загрузка изображения
orig = Image.open(image_path)
bin_img = orig  # можно заменить на бинаризованную версию при необходимости

# === Измеряем время инференса
start = time.perf_counter()
page = pipeline(bin_img)
elapsed = time.perf_counter() - start

# вывод результата и времени
print(page.to_plain_text())
print(f"[INFO] Inference took {elapsed:.3f} seconds")

# визуализация
vis = pipeline.visualize(bin_img)
vis.show()
