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


class Text2Line:
    def __init__(
        self,
        skew_correction: bool = True,
        max_angle: float = 15.0,
        max_iter: int = 100,
        tol=1e-4,
    ):
        self.skew_correction = skew_correction
        self.max_angle = max_angle
        self.max_iter = max_iter
        self.tol = tol
        self.image_width = None
        self.image_height = None

    def get_lines(
        self, text_boxes: List[List[int]], image_width: int, image_height: int
    ):
        """
        Processes a list of text boxes to group them into lines.

        Parameters:
        text_boxes (List[List[int]]): A list of text boxes where each box is represented by a list of four integers:
                                      [x_min, x_max, y_min, y_max]
                                      - x_min: The minimum x-coordinate (left side) of the box.
                                      - x_max: The maximum x-coordinate (right side) of the box.
                                      - y_min: The minimum y-coordinate (top side) of the box.
                                      - y_max: The maximum y-coordinate (bottom side) of the box.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        """
        self.image_width = image_width
        self.image_height = image_height

        sorted_lines_id = self.optimize_skew(text_boxes)

        return sorted_lines_id

    def sort_and_group_boxes(
        self, text_boxes: np.ndarray
    ) -> Tuple[List[List[np.ndarray]], List[List[int]]]:
        """
        Sorts text boxes into lines based on their vertical and horizontal alignment and groups them into lines.

        Parameters:
        text_boxes (np.ndarray): A numpy array of text boxes where each box is represented by four points:
                                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns:
        Tuple: A list of lines, where each line is a list of sorted text boxes and a list of indices corresponding to each box's position in the original list.
        """

        def get_box_center(box):
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
            return x_center, y_center

        # Sort boxes by the y-center
        indexed_boxes = [(idx, box) for idx, box in enumerate(text_boxes)]
        indexed_boxes.sort(key=lambda item: get_box_center(item[1])[1])

        combined_list, index_list = [], []
        new_box, new_index = [], []
        b_height, b_ycenter = [], []

        for idx, box in indexed_boxes:
            _, y_center = get_box_center(box)
            height = max(box[:, 1]) - min(box[:, 1])

            if not new_box:
                b_height, b_ycenter = [height], [y_center]
                new_box.append(box)
                new_index.append(idx)
            elif abs(np.mean(b_ycenter) - y_center) < 0.5 * np.mean(b_height):
                b_height.append(height)
                b_ycenter.append(y_center)
                new_box.append(box)
                new_index.append(idx)
            else:
                combined_list.append(new_box)
                index_list.append(new_index)
                b_height, b_ycenter = [height], [y_center]
                new_box, new_index = [box], [idx]

        if new_box:
            combined_list.append(new_box)
            index_list.append(new_index)

        sorted_combined_list, sorted_index_list = [], []
        for line_boxes, line_indices in zip(combined_list, index_list):
            sorted_line = sorted(
                zip(line_boxes, line_indices), key=lambda item: min(item[0][:, 0])
            )
            sorted_combined_list.extend([box for box, idx in sorted_line])
            sorted_index_list.extend([idx for box, idx in sorted_line])

        grouped_lines, grouped_indices = [], []
        current_line, current_index = [], []
        points = np.array([get_box_center(box) for box in sorted_combined_list])
        min_x_points, max_x_points, mean_x_points = (
            points[:, 0],
            points[:, 1],
            points[:, 0],
        )

        cond = mean_x_points[:-1] > min_x_points[1:]
        cond = np.hstack([[True], cond])

        for box, index, is_new_line in zip(
            sorted_combined_list, sorted_index_list, cond
        ):
            if is_new_line:
                if current_line:
                    grouped_lines.append(current_line)
                    grouped_indices.append(current_index)
                current_line, current_index = [box], [index]
            else:
                current_line.append(box)
                current_index.append(index)

        if current_line:
            grouped_lines.append(current_line)
            grouped_indices.append(current_index)

        return grouped_lines, grouped_indices

    def rotate_boxes(self, boxes: np.ndarray, angle: float) -> np.ndarray:
        angle_rad = np.radians(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        cx, cy = self.image_width / 2, self.image_height / 2

        # Центрирование боксов вокруг центра изображения
        centered_boxes = boxes - np.array([cx, cy])

        # Создание матрицы вращения
        rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        # Вращение всех точек
        rotated_centered_boxes = np.dot(centered_boxes, rotation_matrix.T)

        # Обратное смещение к исходному положению
        rotated_boxes = rotated_centered_boxes + np.array([cx, cy])

        return rotated_boxes

    def compute_angle(self, boxes_by_lines: List[np.ndarray]) -> float:
        angles = []
        for boxes_line in boxes_by_lines:
            if len(boxes_line) < 2:
                continue
            points = np.array(boxes_line).reshape(
                -1, 2
            )  # Преобразование в массив и изменение формы
            _, _, slope = self.calculate_regression_line(points)
            angle = np.arctan(slope) * 180.0 / np.pi
            angles.append(angle)

        if len(angles) == 0:
            return np.inf

        return np.max(np.abs(angles))

    def calculate_regression_line(
        self, points: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        X = points[:, 0]
        y = points[:, 1]
        A = np.vstack([X, np.ones(len(X))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        x_min, x_max = X.min(), X.max()
        y_min, y_max = slope * x_min + intercept, slope * x_max + intercept
        return (x_min, y_min), (x_max, y_max), slope

    def optimize_skew(self, text_boxes):

        def objective_function(angle):
            rotated_boxes = self.rotate_boxes(text_boxes, angle)
            sorted_rotated_lines, _ = self.sort_and_group_boxes(rotated_boxes)
            deviation = self.compute_angle(sorted_rotated_lines)
            return deviation

        best_angle = golden_section_search(
            objective_function, -self.max_angle, self.max_angle, self.tol, self.max_iter
        )

        rotated_boxes = self.rotate_boxes(text_boxes, best_angle)
        _, sorted_id = self.sort_and_group_boxes(rotated_boxes)

        return sorted_id


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
