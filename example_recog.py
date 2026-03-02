from pathlib import Path

import cv2

from recog_tools import extract_text_from_image, load_model

IMAGE_PATH = r"C:\shared\data02065\Archives020525\test_images\558.jpg"
OCR_MODEL_PATH = r"model_095.pth"
CONFIG_PATH = r"config.yaml"
OUTPUT_PATH = None  # absolute or relative path; None -> save to local ./outputs/


def draw_boxes_and_save(image_path: Path, text_results, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    for box, _ in text_results:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save visualization image: {output_path}")


def main() -> None:
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path.resolve()}")

    model = load_model(
        config_path=CONFIG_PATH,
        model_path=OCR_MODEL_PATH,
    )

    # Format: [([x1, y1, x2, y2], "string"), ...]
    text_results = extract_text_from_image(
        image_or_path=str(image_path),
        recognize_model=model,
    )

    result_text = " ".join(text for _, text in text_results).strip()

    script_dir = Path(__file__).resolve().parent
    if OUTPUT_PATH:
        output_path = Path(OUTPUT_PATH)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
    else:
        suffix = image_path.suffix if image_path.suffix else ".jpg"
        output_dir = script_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_boxes_red{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    draw_boxes_and_save(image_path, text_results, output_path)

    print("result_text:")
    print(result_text)
    print(f"saved_boxes_image: {output_path.resolve()}")


if __name__ == "__main__":
    main()
