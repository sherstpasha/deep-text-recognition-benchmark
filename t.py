from PIL import Image
from hardocr import DocumentOCRPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "config.yaml"
image_path = r"C:\data_east\images\1105.jpg"
ocr_model_path = r"C:\Users\pasha\OneDrive\Рабочий стол\best_accuracy.pth"

# Инициализируем наш процессор
pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
)

# Загружаем изображение через PIL
pil_img = Image.open(image_path)

# Распознаем изображение
res = pipeline(pil_img)

print(res)

print(res.to_plain_text())
