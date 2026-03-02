import unicodedata
from typing import Dict, List, Sequence


def normalize_text(text: str, lowercase: bool = True, normalize_unicode: str = "NFC") -> str:
    text = "" if text is None else str(text)
    if normalize_unicode:
        text = unicodedata.normalize(normalize_unicode, text)
    if lowercase:
        text = text.lower()
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current

    return previous[-1]


def levenshtein_distance_tokens(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    if seq1 == seq2:
        return 0
    if len(seq1) == 0:
        return len(seq2)
    if len(seq2) == 0:
        return len(seq1)

    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    previous = list(range(len(seq2) + 1))
    for i, t1 in enumerate(seq1, start=1):
        current = [i]
        for j, t2 in enumerate(seq2, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (t1 != t2)
            current.append(min(insertions, deletions, substitutions))
        previous = current

    return previous[-1]


def cer_score(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_distance += levenshtein_distance(pred, ref)
        total_chars += len(ref)
    return float(total_distance / total_chars) if total_chars > 0 else 0.0


def wer_score(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_distance += levenshtein_distance_tokens(pred_words, ref_words)
        total_words += len(ref_words)
    return float(total_distance / total_words) if total_words > 0 else 0.0


def accuracy_score(predictions: List[str], references: List[str]) -> float:
    if not references:
        return 0.0
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return float(correct / len(references))


def evaluate_recognition(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str],
    lowercase: bool = True,
    normalize_unicode: str = "NFC",
) -> Dict[str, object]:
    matched_images = [img for img in ground_truths.keys() if img in predictions]
    missing_predictions = [img for img in ground_truths.keys() if img not in predictions]

    y_true = []
    y_pred = []
    for image_name in matched_images:
        y_true.append(normalize_text(ground_truths[image_name], lowercase, normalize_unicode))
        y_pred.append(normalize_text(predictions[image_name], lowercase, normalize_unicode))

    return {
        "cer": cer_score(y_pred, y_true),
        "wer": wer_score(y_pred, y_true),
        "accuracy": accuracy_score(y_pred, y_true),
        "matched_images": matched_images,
        "missing_predictions": missing_predictions,
    }
