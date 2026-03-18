"""
Автономная функция визуализации OCR-результатов на изображении.
Зависимости: Pillow, pydantic
"""

from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw
from pydantic import BaseModel


# ── Структуры данных ──────────────────────────────────────────────

class WordResponse(BaseModel):
    index: int = 0
    text: str = ""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0


class StringResponse(BaseModel):
    index: int = 0
    words: List[WordResponse]
    x1: int
    y1: int
    x2: int
    y2: int


class BlockResponse(BaseModel):
    index: int = 0
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


# ── Функция визуализации ─────────────────────────────────────────

def visualize(
    pil_img: Image.Image,
    page: PageResponse,
    *,
    show_words: bool = True,
    show_blocks: bool = True,
    connect_strings: bool = True,
    show_numbers: bool = True,
    word_style: Optional[Dict[str, Any]] = None,
    block_style: Optional[Dict[str, Any]] = None,
    line_style: Optional[Dict[str, Any]] = None,
    number_style: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """
    Рисует на копии входного изображения:
    - слова:  outline/fill по word_style
    - блоки:  outline/fill по block_style
    - строки: соединительные линии между соседними словами по line_style
    - цифры порядковые для боксов (если show_numbers=True)

    Параметры:
        pil_img:          исходное PIL-изображение
        page:             PageResponse с результатами OCR
        show_words:       рисовать прямоугольники слов
        show_blocks:      рисовать прямоугольники блоков
        connect_strings:  рисовать линии между словами в строке
        show_numbers:     рисовать порядковые номера слов
        word_style:       стиль для слов   (outline, fill, width)
        block_style:      стиль для блоков  (outline, fill, width)
        line_style:       стиль для линий   (fill, width)
        number_style:     стиль для номеров (fill, font_size, bg_fill)

    Возвращает:
        PIL.Image.Image с нарисованной визуализацией
    """
    # Стили по умолчанию
    word_style = word_style or {
        "outline": (255, 0, 0, 255),
        "fill": (255, 0, 0, 80),
        "width": 1,
    }
    block_style = block_style or {
        "outline": (0, 0, 255, 255),
        "fill": (0, 0, 255, 50),
        "width": 3,
    }
    line_style = line_style or {"fill": (0, 255, 0, 200), "width": 5}
    number_style = number_style or {
        "fill": (0, 0, 0, 255),
        "font_size": 30,
        "bg_fill": (255, 255, 255, 128),
    }

    # Подготовка холстов
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Рисуем блоки
    if show_blocks:
        for block in page.blocks:
            fill = block_style.get("fill")
            if fill is not None:
                fill = (*fill[:3], fill[3]) if len(fill) == 4 else (*fill, 100)
            draw.rectangle(
                [block.x1, block.y1, block.x2, block.y2],
                outline=block_style["outline"],
                width=block_style["width"],
                fill=fill,
            )

    # Рисуем слова
    if show_words:
        for block in page.blocks:
            for string in block.strings:
                words = sorted(string.words, key=lambda w: w.x1)
                for i, word in enumerate(words):
                    word.index = i + 1
                    fill = word_style.get("fill")
                    if fill is not None:
                        fill = (
                            (*fill[:3], fill[3]) if len(fill) == 4 else (*fill, 100)
                        )
                        draw.rectangle(
                            [word.x1, word.y1, word.x2, word.y2],
                            fill=fill,
                            outline=None,
                        )
                    if word_style.get("outline"):
                        draw.rectangle(
                            [word.x1, word.y1, word.x2, word.y2],
                            outline=word_style["outline"],
                            width=word_style["width"],
                            fill=None,
                        )

    # Соединительные линии
    if connect_strings:
        for block in page.blocks:
            for string in block.strings:
                words = sorted(string.words, key=lambda w: w.x1)
                for prev, curr in zip(words, words[1:]):
                    y_prev = (prev.y1 + prev.y2) // 2
                    y_curr = (curr.y1 + curr.y2) // 2
                    draw.line(
                        [(prev.x2, y_prev), (curr.x1, y_curr)],
                        fill=line_style["fill"],
                        width=line_style["width"],
                    )

    # Порядковые номера
    if show_numbers:
        for block in page.blocks:
            for string in block.strings:
                for word in string.words:
                    text_number = str(word.index)
                    text_size = number_style.get("font_size", 30)
                    text_x = (word.x1 + word.x2) / 2
                    text_y = (word.y1 + word.y2) / 2
                    bg_width = text_size * len(text_number)
                    bg_height = text_size
                    draw.rectangle(
                        [
                            text_x - bg_width / 2,
                            text_y - bg_height / 2,
                            text_x + bg_width / 2,
                            text_y + bg_height / 2,
                        ],
                        fill=number_style["bg_fill"],
                    )
                    draw.text(
                        (text_x - text_size / 2, text_y - text_size / 2),
                        text_number,
                        fill=number_style["fill"],
                    )

    # Комбинируем и возвращаем результат
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


# ── Использование ─────────────────────────────────────────────────
#
#   from visualize_boxes import visualize, PageResponse
#   page = PageResponse(**data)  # данные из OCR
#   vis = visualize(image, page)
#   vis.show()
