from typing import List, Tuple
import numpy as np
import os
from PIL import Image
import re


def golden_section_search(f, a, b, tol=1e-5, max_iter=100):
    gr = (1 + np.sqrt(5)) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(max_iter):
        if abs(c - d) < tol:
            break

        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def convert_boxes_to_points(text_boxes: List[List[int]]) -> List[List[List[int]]]:
    """
    Converts text boxes from format [x_min, x_max, y_min, y_max] to format [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]].

    Parameters:
    text_boxes (List[List[int]]): A list of text boxes where each box is represented by a list of four integers:
                                      [x_min, x_max, y_min, y_max]

    Returns:
    List[List[List[int]]]: A list of text boxes where each box is represented by a list of four lists:
                                     [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    """
    formatted_text_boxes = []
    for box in text_boxes:
        x_min, x_max, y_min, y_max = [max(0, coord) for coord in box]
        formatted_box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        formatted_text_boxes.append(formatted_box)
    return formatted_text_boxes


def assign_boxes_to_lines(
    text_boxes: List[np.ndarray], index_list: List[int]
) -> List[List[np.ndarray]]:
    """
    Assigns text boxes to lines using the provided index list.

    Parameters:
    text_boxes (List[np.ndarray]): A list of text boxes where each box is represented by
                                   an array of four points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    index_list (List[int]): A list of indices corresponding to each box's position in the original list.

    Returns:
    List[List[np.ndarray]]: A list of lines, where each line is a list of sorted text boxes.
    """
    lines = [[] for _ in range(len(index_list))]
    for idx, line_indices in enumerate(index_list):
        for box_idx in line_indices:
            lines[idx].append(text_boxes[box_idx])
    return lines


def merge_boxes_by_lines(grouped_lines):
    """
    Объединяет боксы каждой строки в один бокс путем поиска крайних точек.

    Parameters:
    grouped_lines (List[List[np.ndarray]]): Список строк с боксами, где каждый бокс представлен numpy array с координатами четырех точек.

    Returns:
    List[List[int]]: Список объединенных боксов для каждой строки в формате [x_min, y_min, x_max, y_max].
    """
    merged_boxes = []

    for line in grouped_lines:
        x_min = min([box[:, 0].min() for box in line])
        x_max = max([box[:, 0].max() for box in line])
        y_min = min([box[:, 1].min() for box in line])
        y_max = max([box[:, 1].max() for box in line])
        merged_boxes.append([x_min, y_min, x_max, y_max])

    return merged_boxes


def crop_boxes(image_np, polygons):
    """
    Вырезает части изображения по bounding box координатам из полигонов.

    Parameters:
    image_np (numpy.ndarray): Входное изображение в формате numpy array.
    polygons (list): Список полигонов, каждый из которых представлен списком точек [[x1, y1], [x2, y2], ...].

    Returns:
    list: Список списков вырезанных частей изображения в формате PIL Image, в той же структуре, что и входные полигоны.
    list: Список списков bounding box координат [(x_min, y_min, x_max, y_max), ...] в той же структуре, что и входные полигоны.
    """
    cropped_images_grouped = []

    for group in polygons:
        cropped_images = []
        boxes = []
        for polygon in group:
            # Преобразуем координаты полигона в целые числа
            polygon = [(int(x), int(y)) for x, y in polygon.tolist()]

            # Обрезаем по минимальной и максимальной границам полигона
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            bbox = (x_min, y_min, x_max, y_max)
            boxes.append(bbox)

            # Вырезаем изображение по bounding box
            cropped_image_np = image_np[y_min:y_max, x_min:x_max]

            # Преобразуем вырезанное изображение в формат PIL
            cropped_image_pil = Image.fromarray(cropped_image_np.astype("uint8"))
            cropped_images.append(cropped_image_pil)

        cropped_images_grouped.append(cropped_images)

    return cropped_images_grouped


def save_cropped_images_grouped(cropped_images_grouped, save_dir):
    """
    Сохраняет вырезанные изображения в указанную директорию, сохраняя структуру группировки.

    Parameters:
    cropped_images_grouped (list): Список списков вырезанных изображений в формате PIL Image.
    save_dir (str): Путь к директории для сохранения вырезанных изображений.

    Returns:
    None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, group in enumerate(cropped_images_grouped):
        group_dir = os.path.join(save_dir, f"group_{i}")
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        for j, cropped_image in enumerate(group):
            save_path = os.path.join(group_dir, f"cropped_image_{j}.png")
            cropped_image.save(save_path)


def crop_merged_boxes(image_np, merged_polygons):
    """
    Вырезает части изображения по объединенным bounding box координатам из полигонов.

    Parameters:
    image_np (numpy.ndarray): Входное изображение в формате numpy array.
    merged_polygons (list): Список объединенных bounding box координат [(x_min, y_min, x_max, y_max), ...].

    Returns:
    list: Список вырезанных частей изображения в формате PIL Image.
    """
    cropped_images = []
    height, width = image_np.shape[:2]

    for bbox in merged_polygons:
        x_min, y_min, x_max, y_max = bbox

        # Проверяем границы
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # Убедимся, что bounding box валиден после обрезки
        # if x_min < x_max and y_min < y_max:
        # Вырезаем изображение по bounding box
        cropped_image_np = image_np[y_min:y_max, x_min:x_max]

        # Преобразуем вырезанное изображение в формат PIL
        cropped_image_pil = Image.fromarray(cropped_image_np.astype("uint8"))
        cropped_images.append(cropped_image_pil)

    return cropped_images


def save_merged_cropped_images(cropped_images, save_dir):
    """
    Сохраняет вырезанные изображения в указанную директорию.

    Parameters:
    cropped_images (list): Список вырезанных изображений в формате PIL Image.
    save_dir (str): Путь к директории для сохранения вырезанных изображений.

    Returns:
    None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, cropped_image in enumerate(cropped_images):
        save_path = os.path.join(save_dir, f"cropped_image_{i}.png")
        cropped_image.save(save_path)


def convert_to_integers(grouped_boxes):
    """
    Преобразует все координаты боксов в целые числа.

    Parameters:
    grouped_boxes (List[List[np.ndarray]]): Список строк с боксами, где каждый бокс представлен numpy array с координатами четырех точек.

    Returns:
    List[List[np.ndarray]]: Список строк с боксами, где все координаты представлены целыми числами.
    """
    converted_boxes = []

    for line in grouped_boxes:
        converted_line = []
        for box in line:
            converted_box = np.array(box, dtype=int)
            converted_line.append(converted_box)
        converted_boxes.append(converted_line)

    return converted_boxes


def remove_duplicates(word):
    """
    Удаляет повторяющиеся символы и пары символов в конце слова,
    а также удаляет 3 и более повтора внутри слова.

    Parameters:
    word (str): Входное слово для обработки.

    Returns:
    str: Обработанное слово.
    """
    # Удаляем повторы одного символа в конце слова
    word = re.sub(r"(.)\1{1,}$", r"\1", word)
    # Удаляем повторы пар символов в конце слова
    word = re.sub(r"(.{2})\1+$", r"\1", word)
    # Удаляем 3 и более повтора одного символа внутри слова
    word = re.sub(r"(.)\1{2,}", r"\1\1", word)
    # Удаляем повторяющиеся пары символов внутри слова
    word = re.sub(r"(.{2})\1{1,}", r"\1", word)
    return word


def clean_recognized_text(recognized_text):
    """
    Удаляет все вхождения '[s]' из переданной строки,
    а затем удаляет повторяющиеся символы и пары символов в конце слов,
    а также удаляет 3 и более повтора внутри слов.

    Parameters:
    recognized_text (str): Строка с распознанным текстом.

    Returns:
    str: Очищенная строка.
    """
    # Удаление всех [s] из строки
    recognized_text = re.sub(r"\[s\].*", "", recognized_text)

    # Разбиваем текст на слова
    words = recognized_text.split()
    # Применяем функцию remove_duplicates к каждому слову
    processed_words = [remove_duplicates(word) for word in words]
    # Объединяем обработанные слова обратно в строку
    cleaned_text = " ".join(processed_words)

    return cleaned_text
