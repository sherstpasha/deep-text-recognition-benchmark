import gradio as gr
from PIL import Image, ImageOps, ImageDraw
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


# Исправленная функция рисования боксов поверх текста
def draw_boxes_on_image(image, boxes):
    image_copy = image.copy()  # Создаем копию изображения
    draw = ImageDraw.Draw(image_copy)

    for polygon in boxes:
        for polygon_i in polygon:
            polygon_coords = [(int(x), int(y)) for x, y in polygon_i]
            draw.polygon(polygon_coords, outline="red", width=3)

    return image_copy  # Возвращаем копию изображения с нарисованными боксами


# Функция обработки рисунка с холста
def process_drawing(drawing, gallery_images):
    if gallery_images is None:
        gallery_images = []  # Инициализация пустого списка, если это первое использование

    # Извлечение изображения с холста
    drawing_array = np.array(drawing["composite"], dtype=np.uint8)
    image = Image.fromarray(drawing_array)

    # Преобразование в RGB, если изображение имеет альфа-канал (RGBA)
    if image.mode == "RGBA":
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])
        image = white_background

    # Сохранение оригинального изображения
    saved_image_path = save_drawing(image, prefix="submitted")

    # Преобразуем изображение в массив для распознавания
    image_array = np.array(image)
    try:
        # Здесь модель распознает текст на оригинальном изображении
        results, final_boxes, _ = extract_text_from_image(image_array, model_dict)
        

        if not results:
            return "Текст не найден.", gallery_images, gr.update(visible=False)

        # Объединение распознанного текста
        recognized_text = " ".join([text for _, text in results])

        # Коррекция текста с помощью Яндекс Спеллера
        corrected_text = correct_text(recognized_text)

        # Если есть боксы, рисуем их на изображении
        if final_boxes is not None:
            image_with_boxes = draw_boxes_on_image(image, final_boxes)
            # Сохранение изображения с боксами для галереи
            saved_image_with_boxes_path = save_drawing(image_with_boxes, prefix="with_boxes")

            # Добавляем новое изображение в галерею
            gallery_images.append((saved_image_with_boxes_path, corrected_text))
            return corrected_text, gallery_images, gr.update(visible=True)

        return corrected_text, gallery_images, gr.update(visible=True)
    except ValueError as e:
        return "Текст не найден.", gallery_images, gr.update(visible=False)


# Функция для очистки холста
def clear_drawing():
    return None, [], gr.update(visible=False)


# Gradio интерфейс
with gr.Blocks() as demo:
    # Окно для результата распознавания
    recognized_text = gr.Textbox(
        label="Результат распознавания", interactive=False, elem_id="result-box"
    )

    # Холст для рисования
    canvas = gr.Sketchpad(
        label="Нарисуйте текст", width=1920, height=640, canvas_size=(1920, 780)
    )

    # Кнопки для действий
    with gr.Row():
        recognize_button = gr.Button("Распознать")
        clear_button = gr.Button("Очистить")

    # Галерея для отображения последних распознанных изображений, по умолчанию скрыта
    gallery = gr.Gallery(label="Последние распознанные изображения", columns=4, height="auto", visible=False)

    # Логика кнопок
    recognize_button.click(
        fn=process_drawing, inputs=[canvas, gallery], outputs=[recognized_text, gallery, gallery]
    )  # Распознавание текста и обновление галереи

    clear_button.click(fn=clear_drawing, outputs=[canvas, gallery, gallery])  # Очистка холста и галереи

# Запуск интерфейса
demo.launch(share=True)
