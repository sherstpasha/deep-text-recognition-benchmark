import numpy as np
from PIL import Image

# --------------------------------------
# Вынесенные числовые параметры для reader.detect()
# --------------------------------------
DETECT_PARAMS = {
    "text_threshold": 0.5213592572934042,
    "low_text": 0.36159995700141034,
    "link_threshold": 0.3211728225955032,
    "canvas_size": 2560,
    "mag_ratio": 2.457119400680926,
    "slope_ths": 0.4065218089501456,
    "ycenter_ths": 0.0,
    "height_ths": 0.7308397742828456,
    "width_ths": 0.6589587170590814,
    "add_margin": 0.07212504180872058,
}

# — пример использования —
from PIL import Image
from hardocr import DocumentOCRPipeline
import torch

# настройка устройства, модели и конфига
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "config.yaml"
ocr_model_path = r"C:\Users\pasha\OneDrive\Рабочий стол\best_accuracy.pth"
image_path = r"C:\data0205\Archives020525\test_images\3686.jpg"

# Создание пайплайна с вынесенными detect-параметрами
pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
    detect_params=DETECT_PARAMS,
)

# загрузка изображения
orig = Image.open(image_path)

# используем оригинал без бинаризации
bin_img = orig

# распознаём текст и визуализируем
page = pipeline(bin_img)
print(page.to_plain_text())
vis = pipeline.visualize(bin_img)
vis.show()
