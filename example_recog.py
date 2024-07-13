from recog_tools import load_model, extract_text_from_image


image_path = r"/workspace/mounted_folder/test_img/photo_2024-07-03_19-49-58.jpg"
ocr_model_path = r"/workspace/mounted_folder/best_accuracy4.pth"

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
print(text_results)
