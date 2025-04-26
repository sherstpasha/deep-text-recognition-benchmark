import re
import argparse
import yaml
import torch
import warnings
from typing import List, Tuple, Any, Dict, Union
import numpy as np
from PIL import Image
from easyocr import Reader
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from torchvision import transforms
from pydantic import BaseModel
from PIL import ImageDraw
import cv2
import os


class WordResponse(BaseModel):
    index: int
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


class StringResponse(BaseModel):
    index: int
    words: List[WordResponse]
    x1: int
    y1: int
    x2: int
    y2: int


class BlockResponse(BaseModel):
    index: int
    strings: List[StringResponse]
    x1: int
    y1: int
    x2: int
    y2: int


class PageResponse(BaseModel):
    blocks: List[BlockResponse]

    def word_count(self) -> int:
        return sum(
            len(string.words) for block in self.blocks for string in block.strings
        )

    def to_plain_text(self, line_sep: str = "\n", block_sep: str = "\n\n\n") -> str:
        block_texts: List[str] = []
        for block in self.blocks:
            lines = [" ".join(w.text for w in string.words) for string in block.strings]
            block_texts.append(line_sep.join(lines))
        return block_sep.join(block_texts)


class DocumentOCRPipeline:
    def __init__(
        self,
        config_path: str,
        ocr_model_path: str,
        *,
        device: Union[str, torch.device] = None,
        reader_params: Dict[str, Any] = None,
        max_splits: int = None,
        y_tol_ratio: float = 0.6,
        x_gap_ratio: float = 2.5,
    ):
        """
        :param config_path: путь к YAML-конфигу модели
        :param ocr_model_path: путь к весам .pth
        :param device: "cuda" или "cpu" или torch.device — для PyTorch и EasyOCR
        :param reader_params: доп. параметры для EasyOCR.Reader (кроме 'gpu')
        """
        # 1) определяем self.device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )

        # 2) загружаем модель на self.device
        self.model, self.converter, self.opt = self._load_model(
            config_path, ocr_model_path
        )

        # 3) настраиваем EasyOCR с учётом self.device
        default_params = {"lang_list": ["ru"]}
        if reader_params:
            default_params.update(reader_params)
        # проверяем конфликт по ключу 'gpu'
        if "gpu" in default_params:
            requested = bool(default_params["gpu"])
            actual = self.device.type != "cpu"
            if requested != actual:
                warnings.warn(
                    f"reader_params['gpu']={requested} конфликтует с device={self.device}; "
                    f"используем device={self.device}."
                )
        reader_kwargs = default_params.copy()
        # принудительно выставляем gpu в соотв. device
        reader_kwargs["gpu"] = self.device.type != "cpu"
        # убираем возможность override recognizer
        reader_kwargs.pop("recognizer", None)
        self.reader = Reader(**reader_kwargs, recognizer=None)

        # 4) параметры колонок / строк
        self.max_splits = max_splits
        self.y_tol_ratio = y_tol_ratio
        self.x_gap_ratio = x_gap_ratio

    def _load_model(self, config_path: str, model_path: str):
        """
        Загружает конфигурацию и веса модели.
        :param config_path: путь к YAML конфигу модели
        :param model_path: путь к файлу .pth с весами
        :return: tuple (model, converter, opt)
        """
        # 1. Загружаем опции из конфига
        with open(config_path, "r", encoding="utf-8") as f:
            opt = argparse.Namespace(**yaml.safe_load(f))

        # 2. Выбираем конвертер
        if "CTC" in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)
        if opt.rgb:
            opt.input_channel = 3

        # 3. Строим модель и переносим её на нужное устройство
        model = Model(opt).to(self.device)

        # 4. На GPU с несколькими картами — оборачиваем в DataParallel
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # 5. Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # 6. Если чекпоинт — словарь с ключом "model", берём его, иначе — весь словарь
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # 7. Убираем префикс "module." из имён, если он есть
        cleaned_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned_state[k[len("module.") :]] = v
            else:
                cleaned_state[k] = v

        # 8. Загружаем веса в модель
        model.load_state_dict(cleaned_state, strict=True)
        model.eval()

        return model, converter, opt

    def recognize_word(
        self, pil_img: Image.Image, box: Tuple[int, int, int, int], word_idx: int
    ) -> WordResponse:
        """
        Распознаёт один бокс:
        1) Вырезает ROI из PIL-картинки
        2) Применяет модель + очищает текст
        3) Возвращает WordResponse

        :param pil_img: исходное PIL.Image
        :param box: кортеж (x0, y0, x1, y1)
        :param word_idx: порядковый номер этого слова в строке
        """
        x0, y0, x1, y1 = box
        crop = pil_img.crop((x0, y0, x1, y1))

        raw = self._recognize_text(crop)
        cleaned = self._clean_text(raw)
        return WordResponse(index=word_idx, text=cleaned, x1=x0, y1=y0, x2=x1, y2=y1)

    def visualize(
        self,
        pil_img: Image.Image,
        *,
        show_words: bool = True,
        show_blocks: bool = True,
        connect_strings: bool = True,
        word_style: Dict[str, Any] = None,
        block_style: Dict[str, Any] = None,
        line_style: Dict[str, Any] = None,
    ) -> Image.Image:
        """
        Рисует на копии входного изображения:
          - слова: outline/fill по word_style
          - блоки: outline/fill по block_style
          - строки: соединительные линии между соседними словами по line_style

        :param connect_strings: если True — рисовать линии между словами одной строки
        :param line_style: dict с 'fill' (цвет RGBA) и 'width' для линий строк
        """
        # 1) стили по умолчанию
        word_style = word_style or {"outline": (255, 0, 0, 255), "fill": None, "width": 1}
        block_style = block_style or {"outline": (0, 0, 255, 255), "fill": None, "width": 3}
        line_style = line_style or {"fill": (0, 255, 0, 200), "width": 5}

        # 2) получаем структуру PageResponse
        page = self(pil_img)

        # 3) подготовка холстов
        base = pil_img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 4) рисуем блоки
        if show_blocks:
            for block in page.blocks:
                draw.rectangle(
                    [block.x1, block.y1, block.x2, block.y2],
                    outline=block_style["outline"],
                    width=block_style["width"],
                    fill=block_style.get("fill"),
                )

        # 5) рисуем слова
        if show_words:
            for block in page.blocks:
                for string in block.strings:
                    for w in string.words:
                        if word_style.get("fill"):
                            draw.rectangle(
                                [w.x1, w.y1, w.x2, w.y2],
                                fill=word_style["fill"],
                                outline=None,
                            )
                        if word_style.get("outline"):
                            draw.rectangle(
                                [w.x1, w.y1, w.x2, w.y2],
                                outline=word_style["outline"],
                                width=word_style["width"],
                                fill=None,
                            )

        # 6) рисуем соединительные линии для строк
        if connect_strings:
            for block in page.blocks:
                for string in block.strings:
                    # сортируем слова по x1
                    words = sorted(string.words, key=lambda w: w.x1)
                    for prev, curr in zip(words, words[1:]):
                        # точки: (x2_prev, center_y_prev) → (x1_curr, center_y_curr)
                        y_prev = (prev.y1 + prev.y2) // 2
                        y_curr = (curr.y1 + curr.y2) // 2
                        draw.line(
                            [(prev.x2, y_prev), (curr.x1, y_curr)],
                            fill=line_style["fill"],
                            width=line_style["width"],
                        )

        # 7) комбинируем и возвращаем
        result = Image.alpha_composite(base, overlay)
        return result.convert("RGB")

    def __call__(self, pil_img: Image.Image) -> PageResponse:

        # 1) переводим в RGB и в numpy
        rgb_img = pil_img.convert("RGB")
        img_array = np.array(rgb_img)

        # 2) detect → merge → segment
        h_boxes, f_boxes = self.reader.detect(
            img_array,
            optimal_num_chars=25,
            link_threshold=0.3,
            canvas_size=1280,
        )
        print(f_boxes)
        all_boxes = self._merge_boxes(h_boxes, f_boxes)
        columns = self._segment_columns(all_boxes)

        # 3) ваш остальной код без изменений…
        blocks: List[BlockResponse] = []
        for b_idx, col in enumerate(columns):
            sorted_boxes = self._sort_boxes_reading_order(col)
            lines = self._split_into_lines(sorted_boxes)

            strings: List[StringResponse] = []
            for s_idx, line_boxes in enumerate(lines):
                words = [
                    self.recognize_word(rgb_img, box, w_idx)
                    for w_idx, box in enumerate(line_boxes)
                ]

                xs1 = [w.x1 for w in words]
                ys1 = [w.y1 for w in words]
                xs2 = [w.x2 for w in words]
                ys2 = [w.y2 for w in words]
                strings.append(
                    StringResponse(
                        index=s_idx,
                        words=words,
                        x1=min(xs1),
                        y1=min(ys1),
                        x2=max(xs2),
                        y2=max(ys2),
                    )
                )

            xs1 = [s.x1 for s in strings]
            ys1 = [s.y1 for s in strings]
            xs2 = [s.x2 for s in strings]
            ys2 = [s.y2 for s in strings]
            blocks.append(
                BlockResponse(
                    index=b_idx,
                    strings=strings,
                    x1=min(xs1),
                    y1=min(ys1),
                    x2=max(xs2),
                    y2=max(ys2),
                )
            )

        return PageResponse(blocks=blocks)

    def _split_into_lines(
        self, boxes: List[Tuple[int, int, int, int]], x_tol: int = 0
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Разбивает отсортированный список боксов на строки.
        Начало новой строки, если x1 текущего < x1 предыдущего - x_tol.
        """
        if not boxes:
            return []
        lines = [[boxes[0]]]
        prev_x0 = boxes[0][0]
        for b in boxes[1:]:
            x0 = b[0]
            if x0 + x_tol < prev_x0:
                # новая строка
                lines.append([b])
            else:
                lines[-1].append(b)
            prev_x0 = x0
        return lines

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Преобразует PIL изображение в тензор для распознавания.
        """
        # Оригинальная предобработка через transforms
        if not self.opt.rgb:
            image = image.convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((self.opt.imgH, self.opt.imgW), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        img = transform(image).unsqueeze(0)  # Add batch dimension
        return img.to(self.device)

    def _recognize_text(self, image: Image.Image) -> str:
        tensor = self._preprocess(image)
        batch_size = 1
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(
            self.device
        )
        text_for_pred = (
            torch.LongTensor(batch_size, self.opt.batch_max_length + 1)
            .fill_(0)
            .to(self.device)
        )
        with torch.no_grad():
            if "CTC" in self.opt.Prediction:
                preds = self.model(tensor, text_for_pred)
                _, indices = preds.max(2)
                raw = self.converter.decode(
                    indices.data, torch.IntTensor([preds.size(1)])
                )
            else:
                preds = self.model(tensor, text_for_pred, is_train=False)
                _, indices = preds.max(2)
                raw = self.converter.decode(indices, torch.IntTensor([batch_size]))
        return raw[0]

    def _remove_duplicates(self, word: str) -> str:
        if len(word) <= 3:
            return word
        word = re.sub(r"(.)\1{1,}$", r"\1", word)
        word = re.sub(r"(.{2})\1+$", r"\1", word)
        word = re.sub(r"(.)\1{2,}", r"\1\1", word)
        word = re.sub(r"(.{2})\1{1,}", r"\1", word)
        return word

    def _clean_text(self, text: str) -> str:
        txt = re.sub(r"\[s\].*", "", text)
        return " ".join([self._remove_duplicates(w) for w in txt.split()])

    def _merge_boxes(
        self, h_boxes: List[Any], f_boxes: List[Any]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Объединяет боксы из h_boxes и f_boxes в список прямоугольников (x0, y0, x1, y1).
        Обрезает все отрицательные x0, y0 → 0, чтобы не было «выпавших» боксов.
        """
        raw_h = h_boxes[0] if h_boxes and isinstance(h_boxes[0], list) else []
        raw_f = f_boxes[0] if f_boxes and isinstance(f_boxes[0], list) else []
        merged: List[Tuple[int, int, int, int]] = []

        # горизонтальные прямоугольники
        for box in raw_h:
            x0 = max(0, int(box[0]))
            y0 = max(0, int(box[2]))
            x1 = int(box[1])
            y1 = int(box[3])
            merged.append((x0, y0, x1, y1))

        # полигональные, приводим к axis-aligned
        for poly in raw_f:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            x0 = max(0, int(min(xs)))
            y0 = max(0, int(min(ys)))
            x1 = int(max(xs))
            y1 = int(max(ys))
            merged.append((x0, y0, x1, y1))

        return merged

    def _find_gaps(self, boxes, start, end) -> List[int]:
        # собираем все перекрытия внутри [start, end]
        segs = [
            (max(b[0], start), min(b[2], end))
            for b in boxes
            if not (b[2] <= start or b[0] >= end)
        ]
        if not segs:
            return []
        segs.sort()
        # объединяем пересекающиеся
        merged = [segs[0]]
        for s, e in segs[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        # находим пробелы между ними
        gaps = []
        prev_end = start
        for s, e in merged:
            if s > prev_end:
                gaps.append((prev_end, s))
            prev_end = e
        if prev_end < end:
            gaps.append((prev_end, end))
        # возвращаем центры «пустых» областей
        return [(a + b) // 2 for a, b in gaps if b - a > 1]

    def _emptiness(self, boxes, start, end) -> float:
        col = [b for b in boxes if b[0] >= start and b[2] <= end]
        if not col:
            return 1.0
        min_y, min_x = min(b[1] for b in col), max(b[3] for b in col)
        rect = (end - start) * (min_x - min_y)
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in col)
        return (rect - area) / rect if rect else 1.0

    def _segment_columns(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Разбивает боксы на колонки по «пустым» вертикальным промежуткам.
        В конце отфильтровывает пустые колонки перед сортировкой.
        """
        if not boxes:
            return []

        img_width = max(b[2] for b in boxes)
        segments = [(0, img_width)]
        separators: List[int] = []

        # Находим оптимальные разрезы
        for _ in range(self.max_splits or img_width):
            best = None
            for idx, (s, e) in enumerate(segments):
                for x in self._find_gaps(boxes, s, e):
                    if not (
                        any(b[2] <= x and b[0] >= s for b in boxes)
                        and any(b[0] >= x and b[2] <= e for b in boxes)
                    ):
                        continue
                    score = self._emptiness(boxes, s, x) + self._emptiness(boxes, x, e)
                    if best is None or score < best[0]:
                        best = (score, x, idx)
            if not best:
                break
            _, x_split, idx = best
            s, e = segments.pop(idx)
            separators.append(x_split)
            segments.insert(idx, (s, x_split))
            segments.insert(idx + 1, (x_split, e))
            segments.sort()

        # Разбиваем по найденным разделителям
        parts = [(0, img_width)]
        for x in separators:
            new_parts: List[Tuple[int, int]] = []
            for s, e in parts:
                if s < x < e:
                    new_parts += [(s, x), (x, e)]
                else:
                    new_parts.append((s, e))
            parts = new_parts

        # Формируем колонки
        cols: List[List[Tuple[int, int, int, int]]] = []
        for s, e in parts:
            col = [b for b in boxes if b[0] >= s and b[2] <= e]
            cols.append(col)

        # Убираем пустые колонки
        cols = [c for c in cols if c]
        if not cols:
            return []

        # Сортируем по x-координате левого края
        return sorted(cols, key=lambda c: min(b[0] for b in c))

    def _sort_boxes_reading_order(self, boxes) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        avg_h = np.mean([b[3] - b[1] for b in boxes])
        lines = []
        for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
            cy = (b[1] + b[3]) / 2
            placed = False
            for ln in lines:
                line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
                if (
                    abs(cy - line_cy) <= avg_h * self.y_tol_ratio
                    and b[0] - max(v[2] for v in ln) < avg_h * self.x_gap_ratio
                ):
                    ln.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])
        lines.sort(key=lambda ln: np.mean([(v[1] + v[3]) / 2 for v in ln]))
        for ln in lines:
            ln.sort(key=lambda v: v[0])
        return [v for ln in lines for v in ln]
