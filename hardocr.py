import argparse
import yaml
import torch
from typing import List, Tuple, Any, Dict
import numpy as np
from easyocr import Reader
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter

# Устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config_path: str, model_path: str):
    """
    Загружает конфигурацию и веса модели.
    :param config_path: путь к YAML конфигу модели
    :param model_path: путь к файлу .pth с весами
    :return: tuple (model, converter, opt)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        opt = argparse.Namespace(**yaml.safe_load(f))
    # Выбор конвертера
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    # Создание модели и загрузка весов
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    state = torch.load(model_path, map_location=device)
    if list(state.keys())[0].startswith('module.'):
        model.load_state_dict(state)
    else:
        model.load_state_dict({ 'module.'+k: v for k,v in state.items() })
    model.eval()
    return model, converter, opt


class OCRBoxProcessor:
    def __init__(
        self,
        config_path: str,
        ocr_model_path: str,
        reader_params: Dict[str, Any] = None,
        max_splits: int = None,
        y_tol_ratio: float = 0.6,
        x_gap_ratio: float = 2.5
    ):
        """
        Инициализация: загружает модель и EasyOCR Reader.
        """
        # Загрузка модели через отдельную функцию
        self.model, self.converter, self.opt = load_model(config_path, ocr_model_path)
        # Инициализация EasyOCR Reader
        default_params = {"lang_list": ["ru"]}
        if reader_params:
            default_params.update(reader_params)
                # Инициализация EasyOCR Reader с фиксированным recognizer=None
        # Убираем возможность переопределить recognizer
        reader_kwargs = default_params.copy()
        reader_kwargs.pop('recognizer', None)
        self.reader = Reader(**reader_kwargs, recognizer=None)
        # Параметры сегментации
        self.max_splits = max_splits
        self.y_tol_ratio = y_tol_ratio
        self.x_gap_ratio = x_gap_ratio

    def __call__(
        self,
        image: Any
    ) -> List[Tuple[int, int, int, int]]:
        """
        Полный пайплайн: обнаружение, объединение и сортировка.
        При вызове экземпляра напрямую возвращает отсортированные боксы.
        :param image: путь к изображению или numpy array
        :return: отсортированные боксы (x0, y0, x1, y1)
        """
        h_boxes, f_boxes = self.reader.detect(image)
        all_boxes = self._merge_boxes(h_boxes, f_boxes)
        columns = self._segment_columns(all_boxes)
        ordered: List[Tuple[int,int,int,int]] = []
        for col in columns:
            ordered.extend(self._sort_boxes_reading_order(col))
        return ordered

    def _merge_boxes(
        self,
        h_boxes: List[Any],
        f_boxes: List[Any]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Объединяет боксы из h_boxes и f_boxes в плоский список (x0, y0, x1, y1).
        """
        raw_h = h_boxes[0] if h_boxes and isinstance(h_boxes[0], list) else []
        raw_f = f_boxes[0] if f_boxes and isinstance(f_boxes[0], list) else []
        merged: List[Tuple[int,int,int,int]] = []
        for box in raw_h:
            x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
            merged.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        for poly in raw_f:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            merged.append((x0, y0, x1, y1))
        return merged

    def _find_gaps(
        self,
        boxes: List[Tuple[int,int,int,int]],
        start: int,
        end: int
    ) -> List[int]:
        """Возвращает центры пустых промежутков между s и e."""
        segs = []
        for x0, _, x1, _ in boxes:
            if x1 <= start or x0 >= end:
                continue
            segs.append((max(x0, start), min(x1, end)))
        if not segs:
            return []
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        gaps, prev = [], start
        for s, e in merged:
            if s > prev:
                gaps.append((prev, s))
            prev = max(prev, e)
        if prev < end:
            gaps.append((prev, end))
        return [(a + b) // 2 for a, b in gaps if b - a > 1]

    def _emptiness(
        self,
        boxes: List[Tuple[int,int,int,int]],
        start: int,
        end: int
    ) -> float:
        """Метрика "пустоты" внутри [start, end]."""
        col = [b for b in boxes if b[0] >= start and b[2] <= end]
        if not col:
            return 1.0
        min_y = min(b[1] for b in col)
        max_y = max(b[3] for b in col)
        rect_area = (end - start) * (max_y - min_y)
        boxes_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in col)
        return (rect_area - boxes_area) / rect_area

    def _segment_columns(
        self,
        boxes: List[Tuple[int,int,int,int]]
    ) -> List[List[Tuple[int,int,int,int]]]:
        """Разбивает боксы на колонки по пустым вертикальным промежуткам."""
        if not boxes:
            return []
        img_width = max(b[2] for b in boxes)
        segments, separators = [(0, img_width)], []
        for _ in range(self.max_splits or img_width):
            best = None
            for idx, (s, e) in enumerate(segments):
                for x in self._find_gaps(boxes, s, e):
                    left = any(b[2] <= x and b[0] >= s for b in boxes)
                    right = any(b[0] >= x and b[2] <= e for b in boxes)
                    if not (left and right):
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
            segments.insert(idx+1, (x_split, e))
            segments.sort()
        cols, segs = [], [(0, img_width)]
        for x in separators:
            new, segs2 = [], []
            for s, e in segs:
                if s < x < e:
                    new += [(s, x), (x, e)]
                else:
                    new.append((s, e))
            segs = new
        for s, e in segs:
            cols.append([b for b in boxes if b[0] >= s and b[2] <= e])
        return sorted(cols, key=lambda col: min(b[0] for b in col))

    def _sort_boxes_reading_order(
        self,
        boxes: List[Tuple[int,int,int,int]]
    ) -> List[Tuple[int,int,int,int]]:
        """Сортирует боксы внутри колонки по строкам слева-направо."""
        if not boxes:
            return []
        avg_h = np.mean([y1-y0 for _, y0, _, y1 in boxes])
        lines = []
        for box in sorted(boxes, key=lambda b: (b[1]+b[3])/2):
            x0, y0, x1, y1 = box
            cy = (y0 + y1)/2
            placed = False
            for line in lines:
                line_cy = np.mean([(b[1]+b[3])/2 for b in line])
                if abs(cy-line_cy) <= avg_h*self.y_tol_ratio and x0 - max(b[2] for b in line) < avg_h*self.x_gap_ratio:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])
        lines.sort(key=lambda ln: np.mean([(b[1]+b[3])/2 for b in ln]))
        for ln in lines:
            ln.sort(key=lambda b: b[0])
        return [b for ln in lines for b in ln]