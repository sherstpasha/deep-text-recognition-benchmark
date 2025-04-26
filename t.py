import cv2
import numpy as np
from PIL import Image


def binarize_image(
    pil_img: Image.Image,
    method: str = "otsu",
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    block_size: int = 15,
    c: int = 5,
) -> Image.Image:
    """
    Выполняет предобработку: конвертирует в градации серого и применяет бинаризацию.
    :param pil_img: входной PIL.Image
    :param method: "otsu" или "adaptive"
    :param adaptive_method: cv2.ADAPTIVE_THRESH_MEAN_C или cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    :param block_size: размер окна для adaptive
    :param c: константа C для adaptive
    :return: бинаризованное PIL.Image (L-режим)
    """
    gray = np.array(pil_img.convert("L"))

    if method == "adaptive":
        # adaptiveThreshold требует 8-битное входное, блок_size нечётный
        thresh = cv2.adaptiveThreshold(
            gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size | 1, c
        )
    else:  # otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(thresh)


# — пример использования —

from PIL import Image
from hardocr import DocumentOCRPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "config.yaml"
ocr_model_path = r"C:\shared\saved_models_25_04\TPS-ResNet-BiLSTM-Attn-Seed1111\best_norm_ED.pth"
image_path = r"D:\Archive_19_04\combined_images\9.jpg"

pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
)

# загружаем оригинал
orig = Image.open(image_path)

# препроцессинг
# bin_img = binarize_image(orig, method="otsu")

# bin_img = binarize_image(orig, method="otsu", block_size=31, c=10)
bin_img = orig

# распознаём уже на бинаризованном
# распознаём уже на бинаризованном
page = pipeline(bin_img)
print(page.to_plain_text())

# и визуализируем
vis = pipeline.visualize(bin_img)
vis.show()
