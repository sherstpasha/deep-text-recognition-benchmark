from recog_tools import load_model, detect_image_craft
from easyocr import Reader
from typing import List, Tuple, Any


def merge_detected_boxes(
    h_boxes: List[Any],
    f_boxes: List[Any]
) -> List[Tuple[int, int, int, int]]:
    """
    Объединяет коробки из h_boxes и f_boxes (как в EasyOCR.detect) в один
    плоский список кортежей (x0, y0, x1, y1), не меняя оригиналы.

    - h_boxes: обычно [[ [x_min, x_max, y_min, y_max, ...], ... ]]
    - f_boxes: обычно [[ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ... ]]
    """
    # Вытащим «нулевые» уровни (обычно именно там лежат боксы)
    raw_h = h_boxes[0] if h_boxes and isinstance(h_boxes[0], list) else []
    raw_f = f_boxes[0] if f_boxes and isinstance(f_boxes[0], list) else []

    merged: List[Tuple[int,int,int,int]] = []

    # Сначала h_boxes: [x_min, x_max, y_min, y_max, ...]
    for box in raw_h:
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        merged.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    # Потом f_boxes: полигоны [[x1,y1],...,[x4,y4]]
    for poly in raw_f:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        merged.append((x0, y0, x1, y1))

    return merged

image_path = r"C:\shared\Archive_19_04\combined_images\4.jpg"
ocr_model_path = r"C:\shared\saved_models_23_04\TPS-ResNet-LSTM-Attn-Seed1111\best_accuracy.pth"

reader = Reader(["ru"], recognizer=None)
h_boxes, f_boxes = reader.detect(
        image_path
    )

print(h_boxes)

print(f_boxes)

all_rects  = merge_detected_boxes(h_boxes, f_boxes)

print(all_rects)

'''
# загрузка модели. Словарь с ключами model, converter, opt
model = load_model(
    config_path="config.yaml",  # файл с конфигурацией (не нужно менять)
    model_path=ocr_model_path,  # путь до весов
)

# результат распознавания в формате [([x1, y1, x2, y2], "string"), ...]
text_results = extract_text_from_image(
    image_or_path=image_path,  # Путь к изображению или изображение
    recognize_model=model,  # Путь к модели
)

# Печатаем результаты распознавания текста
print(text_results)'''