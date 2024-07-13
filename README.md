## Использование

### 1. Распознавание текста с помощью предобученной модели

Для распознавания текста на изображении с использованием предобученной модели, выполните следующие шаги:

```python
from recog_tools import load_model, extract_text_from_image

# Пути к изображению и предобученной модели
image_path = r"/workspace/mounted_folder/test_img/photo_2024-07-03_19-49-58.jpg"
ocr_model_path = r"/workspace/mounted_folder/best_accuracy4.pth"

# Загрузка модели. Словарь с ключами: model, converter, opt
model = load_model(
    config_path="config.yaml",  # Файл с конфигурацией (не нужно менять)
    model_path=ocr_model_path,  # Путь до весов модели
)

# Результат распознавания в формате [([x1, y1, x2, y2], "string"), ...]
text_results = extract_text_from_image(
    image_or_path=image_path,  # Путь к изображению или само изображение
    recognize_model=model,  # Загруженная модель
)

# Печать результатов распознавания текста
print(text_results)
```

### 2. Обучение модели

Для обучения модели используйте следующий скрипт:

```python
from train_func import train

train(
    train_csv=r"/workspace/mounted_folder/labels.csv",  # Путь к CSV файлу с метками для обучения
    train_root=r"/workspace/mounted_folder/img",  # Путь к папке с изображениями для обучения
    valid_csv=r"/workspace/mounted_folder/labels.csv",  # Путь к CSV файлу с метками для валидации
    valid_root=r"/workspace/mounted_folder/img",  # Путь к папке с изображениями для валидации
    batch_size=16,  # Размер батча
    num_iter=100,  # Количество итераций
    valInterval=10,  # Интервал валидации
    saved_model="",  # Путь до модели, от которой начинать обучение. Если пусто, то обучение с нуля
    output_dir="test",  # Путь до директории, в которой сохраняется результат
)
```