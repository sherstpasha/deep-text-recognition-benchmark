import gradio as gr
from PIL import Image, ImageOps
import requests
from recog_tools import load_model, extract_text_from_image
import numpy as np
import os
import uuid

# Загрузка модели один раз, чтобы она была доступна глобально
CONFIG_PATH = "config.yaml"
MODEL_PATH = "model_095.pth"
model_dict = load_model(CONFIG_PATH, MODEL_PATH)

# Папка для сохранения отправленных изображений
SAVE_DIR = "submitted_drawings"
os.makedirs(SAVE_DIR, exist_ok=True)


# Функция для коррекции текста с помощью Yandex Speller API
def correct_text(text):
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {"text": text, "lang": "ru"}
    response = requests.get(url, params=params)
    data = response.json()

    corrected_text = text
    for error in reversed(data):
        word = error["word"]
        suggestion = error["s"][0] if error["s"] else word
        corrected_text = (
            corrected_text[: error["pos"]]
            + suggestion
            + corrected_text[error["pos"] + len(word) :]
        )
    return corrected_text


# Функция сохранения рисунков для отладки
def save_drawing(image, save_dir=SAVE_DIR, prefix="drawing"):
    filename = os.path.join(save_dir, f"{prefix}_{uuid.uuid4().hex}.png")
    image.save(filename)
    return filename


# Функция обработки рисунка с холста
def process_drawing(drawing):
    # Извлечение составного изображения (composite) из словаря drawing
    drawing_array = np.array(drawing["composite"], dtype=np.uint8)

    # Преобразование рисунка в изображение PIL
    image = Image.fromarray(drawing_array)

    # Преобразование в RGB, если изображение имеет альфа-канал (RGBA)
    if image.mode == "RGBA":
        # Создание белого фона и наложение рисунка
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(
            image, mask=image.split()[3]
        )  # Используем альфа-канал как маску
        image = white_background

    # Сохранение оригинального изображения для отладки
    saved_image_path = save_drawing(image, prefix="original")

    # Преобразуем изображение в массив numpy
    image_array = np.array(image)

    # Извлечение текста из изображения с обработкой возможных ошибок
    try:
        results = extract_text_from_image(image_array, model_dict)

        if not results:
            return "Текст не найден."

        # Объединение распознанного текста
        recognized_text = " ".join([text for _, text in results])

        # Коррекция текста с помощью Яндекс Спеллера
        corrected_text = correct_text(recognized_text)

        return corrected_text
    except ValueError as e:
        return "Текст не найден."


# Функция для очистки холста
def clear_drawing():
    return None


# Gradio интерфейс
with gr.Blocks() as demo:
    # Окно для результата распознавания
    recognized_text = gr.Textbox(
        label="Результат распознавания", interactive=False, elem_id="result-box"
    )

    # Кнопки для действий
    with gr.Row():
        clear_button = gr.Button("Очистить")
        recognize_button = gr.Button("Распознать")

    # Холст для рисования
    canvas = gr.Sketchpad(
        label="Нарисуйте текст", width=1920, height=1080
    )  # Увеличенный холст для полноэкранного режима

    # Логика кнопок
    clear_button.click(fn=clear_drawing, outputs=canvas)  # Очистка холста
    recognize_button.click(
        fn=process_drawing, inputs=canvas, outputs=recognized_text
    )  # Распознавание текста

# Запуск интерфейса
demo.launch(share=True)
