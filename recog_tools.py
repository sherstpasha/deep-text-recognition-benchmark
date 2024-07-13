from easyocr import Reader
import numpy as np
from tools import (
    convert_boxes_to_points,
    Text2Line,
    assign_boxes_to_lines,
    convert_to_integers,
    merge_boxes_by_lines,
    crop_boxes,
    clean_recognized_text,
)
import os
import yaml
import torch
import argparse
from PIL import Image
from torchvision import transforms
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_image_craft(image, gpu=False):
    # Загрузка изображения
    image_width, image_height = image.shape[0], image.shape[1]

    # Инициализация EasyOCR Reader
    reader = Reader(["ru"], gpu=gpu, recognizer=None)

    # Определение текста с ограничивающими рамками
    h_boxes, f_boxes = reader.detect(
        image,
    )

    h_boxes = h_boxes[0]
    f_boxes = f_boxes[0]
    all_boxes = np.array(convert_boxes_to_points(h_boxes) + f_boxes)

    text2line = Text2Line(max_iter=500)

    sorted_lines_id = text2line.get_lines(all_boxes, image_width, image_height)
    final_boxes = assign_boxes_to_lines(all_boxes, sorted_lines_id)

    return convert_to_integers(final_boxes)


def check_model_device(model):
    """
    Проверяет устройство всех параметров и буферов модели.

    Parameters:
    model (torch.nn.Module): Модель для проверки.
    """
    # Проверка устройства для всех параметров
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")

    # Проверка устройства для всех буферов
    for name, buffer in model.named_buffers():
        print(f"Buffer {name} is on device: {buffer.device}")


def check_tensor_device(tensor):
    """
    Проверяет устройство тензора.

    Parameters:
    tensor (torch.Tensor): Тензор для проверки.
    """
    print(f"Tensor is on device: {tensor.device}")


def load_model(config_path, model_path):
    """Load the trained model"""
    with open(config_path, "r", encoding="utf-8") as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print(f"Loading pretrained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {"model": model, "converter": converter, "opt": opt}


def preprocess_image(image, opt, device):
    """Preprocess the input image"""
    if not opt.rgb:
        image = image.convert("L")

    transform = transforms.Compose(
        [
            transforms.Resize((opt.imgH, opt.imgW), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    img = transform(image).unsqueeze(0)  # Add batch dimension
    return img.to(device)


def recognize_text(model, converter, opt, image):
    """Recognize text from the preprocessed image"""
    image = preprocess_image(image, opt, device)

    length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

    if "CTC" in opt.Prediction:
        preds = model(image, text_for_pred)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, torch.IntTensor([preds.size(1)]))
    else:
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    return preds_str


def recognize_text_from_images(image_pieces, model_dict):

    # Process each image and recognize text
    recognized_texts = []
    for image in image_pieces:
        text = recognize_text(
            model_dict["model"],
            model_dict["converter"],
            model_dict["opt"],
            image,
        )
        recognized_texts.append(clean_recognized_text(text[0]))

    return recognized_texts


def extract_text_from_image(image_or_path, recognize_model):

    gpu = torch.cuda.is_available()

    # Проверяем, является ли входное изображение путем или numpy.ndarray
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError("Невозможно загрузить изображение по указанному пути.")
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise ValueError(
            "image_or_path должен быть либо строкой пути, либо объектом numpy.ndarray"
        )

    final_boxes = detect_image_craft(image, gpu=gpu)
    line_boxes = merge_boxes_by_lines(final_boxes)

    cropped_images_grouped = crop_boxes(np.array(image), final_boxes)

    recognized_text = [
        " ".join(recognize_text_from_images(image_piece, recognize_model))
        for image_piece in cropped_images_grouped
    ]

    # Формируем результат в формате [([x1, y1, x2, y2], "string"), ...]
    results = [
        ([box[0], box[1], box[2], box[3]], text)
        for box, text in zip(line_boxes, recognized_text)
    ]

    return results
