import numpy as np
from PIL import Image

# --------------------------------------
# Вынесенные числовые параметры для reader.detect()
# --------------------------------------
DETECT_PARAMS = {
    "shrink_ratio": 0.6,
    "ycenter_ths": 0.0,
    "iou_threshold": 0.2,
    "score_thresh": 0.9,
    "target_size": 1024,
}


# — пример использования —
from PIL import Image
from hardocr import DocumentOCRPipeline
import torch

# настройка устройства, модели и конфига
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "config.yaml"
ocr_model_path = r"C:\shared\saved_models_25_04\TPS-ResNet-BiLSTM-Attn-Seed1111\best_norm_ED.pth"
image_path = (
    r"C:\shared\data0205\Archives020525\test_images\15.jpg"
)

# Создание пайплайна с вынесенными detect-параметрами
pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
    detect_params=DETECT_PARAMS,
    TTA=True,
    TTA_thresh=0.1,
    use_nms=True,
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
