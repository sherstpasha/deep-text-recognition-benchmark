"""
Автономный скрипт: OCR + визуализация боксов на документе.
Зависимости: Pillow, pydantic, torch, hardocr
Можно перенести в другой проект целиком.
"""

from typing import List, Optional
from PIL import Image, ImageDraw
from pydantic import BaseModel
import torch
import time


# ══════════════════════════════════════════════════════════════════
#  Структуры данных (автономные, не зависят от hardocr)
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
#  Функция визуализации (автономная, не зависит от hardocr)
# ══════════════════════════════════════════════════════════════════

def visualize(
    pil_img: Image.Image,
    page: PageResponse,
    *,
    box_color: tuple = (0, 120, 255, 90),
    outline_color: tuple = (0, 120, 255, 180),
    outline_width: int = 2,
) -> Image.Image:
    """
    Рисует на копии изображения полупрозрачные синие боксы слов.
    """
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for block in page.blocks:
        for string in block.strings:
            for word in string.words:
                draw.rectangle(
                    [word.x1, word.y1, word.x2, word.y2],
                    fill=box_color,
                    outline=outline_color,
                    width=outline_width,
                )

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


# ══════════════════════════════════════════════════════════════════
#  Запуск: OCR → визуализация
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from hardocr import DocumentOCRPipeline
    from pathlib import Path

    # --- настройки ---
    config_path = "config.yaml"
    ocr_model_path = r"best_accuracy.pth"
    images_dir = Path(r"C:\Users\USER\Desktop\images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DocumentOCRPipeline(
        config_path=config_path,
        ocr_model_path=ocr_model_path,
        device=device,
        TTA=True,
        TTA_thresh=0.1,
        use_nms=True,
        batch_size=8,
    )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in exts)
    print(f"[INFO] Найдено {len(files)} изображений в {images_dir}")

    for fpath in files:
        img = Image.open(fpath)

        start = time.perf_counter()
        raw_page = pipeline(img)
        elapsed = time.perf_counter() - start

        page = PageResponse(**raw_page.model_dump())

        out_name = f"{fpath.stem}_var_2.jpg"
        out_path = images_dir / out_name

        vis = visualize(img, page)
        vis.save(str(out_path))
        print(f"[OK] {fpath.name} → {out_name}  ({elapsed:.2f}s, {page.word_count()} words)")
