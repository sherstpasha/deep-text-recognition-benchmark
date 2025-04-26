import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from hardocr import DocumentOCRPipeline
from thefittest.optimizers import DifferentialEvolution

# --------------------------------------
# Границы для параметров детекции и мержа боксов
# --------------------------------------
PARAM_BOUNDS = {
    'text_threshold': (0.1, 1.0),
    'low_text': (0.0, 0.6),
    'link_threshold': (0.0, 1.0),
    'canvas_size': (256, 2048),
    'mag_ratio': (0.5, 3.0),
    'slope_ths': (0.0, 1.0),
    'ycenter_ths': (0.0, 1.0),
    'height_ths': (0.0, 1.0),
    'width_ths': (0.0, 1.0),
    'add_margin': (0.0, 0.5),
}

# --------------------------------------
# Настройка OCR-пайплайна
# --------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = 'config.yaml'
ocr_model_path = r"C:\shared\saved_models_25_04\TPS-ResNet-BiLSTM-Attn-Seed1111\best_norm_ED.pth"
base_pipeline = DocumentOCRPipeline(
    config_path=config_path,
    ocr_model_path=ocr_model_path,
    device=device,
    detect_params={}  # будет задаваться динамически
)

# --------------------------------------
# Метрики: IoU, Dice, Levenshtein
# --------------------------------------
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

def compute_dice(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return 2 * interArea / float(areaA + areaB)

def edit_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1): dp[i][0] = i
    for j in range(len2 + 1): dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[len1][len2]

# --------------------------------------
# Загрузка данных
# --------------------------------------
image_dir = r"C:\shared\Archive_19_04\combined_images"
annotations_csv = r"C:\shared\Archive_19_04\annotations_with_image_size_c.csv"
all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
file_list = all_files[::100]  # каждые 100-е изображение

# --------------------------------------
# Оценка пайплайна
# --------------------------------------
def evaluate_pipeline(pipeline, image_dir, annotations_csv, file_list, iou_thresh=0.5):
    df_ann = pd.read_csv(annotations_csv)
    results = []
    for image_name in file_list:
        path = os.path.join(image_dir, image_name)
        if not os.path.exists(path): continue
        img = Image.open(path)
        page = pipeline(img)
        preds = [((w.x1, w.y1, w.x2, w.y2), w.text)
                 for blk in page.blocks for strn in blk.strings for w in strn.words]
        ann = df_ann[df_ann['image_name'] == image_name]
        if ann.empty: continue
        iw, ih = ann['image_width'].iloc[0], ann['image_height'].iloc[0]
        truths = []
        for _, row in ann.iterrows():
            xc, yc = row['x_center'] * iw, row['y_center'] * ih
            bw, bh = row['box_width'] * iw, row['box_height'] * ih
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            truths.append(((x1, y1, x2, y2), str(row['decoding'])))
        ious, dices = [], []
        correct = total = 0
        norm_eds = []
        used = set()
        for t_box, t_text in truths:
            best_iou, best_pred = 0.0, None
            for idx, (p_box, p_text) in enumerate(preds):
                if idx in used: continue
                iou = compute_iou(t_box, p_box)
                if iou > best_iou:
                    best_iou, best_pred = iou, (idx, p_box, p_text)
            ious.append(best_iou)
            if best_pred and best_iou >= iou_thresh:
                idx, pb, pt = best_pred; used.add(idx)
                dices.append(compute_dice(t_box, pb))
                total += 1
                if pt == t_text: correct += 1
                ed = edit_distance(pt, t_text)
                # инвертируем нормальную ED: 1 - norm_ed
                norm_val = ed / max(len(t_text), 1)
                norm_eds.append(1 - norm_val)
            else:
                dices.append(0.0)
                norm_eds.append(0.0)
        if not ious: continue
        mean_iou = np.mean(ious); mean_dice = np.mean(dices)
        recog_acc = correct / total if total > 0 else 0.0
        # mean_norm_score: чем больше тем лучше
        mean_norm_score = np.mean(norm_eds) if norm_eds else 0.0
        results.append({
            'image': image_name,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'recog_accuracy': recog_acc,
            'mean_norm_score': mean_norm_score,
            'num_true': len(truths),
            'num_pred': len(preds)
        })
    df = pd.DataFrame(results)
    # итоговая метрика: сумма mean_iou + recog_accuracy + mean_norm_score
    df['total_score'] = df['mean_iou'] + df['recog_accuracy'] + df['mean_norm_score']
    return df

# --------------------------------------
# Сохранение базовой статистики перед оптимизацией
# --------------------------------------
HISTORY_FILE = 'de_history.txt'
BASELINE_FILE = 'baseline_stats.txt'

def save_baseline():
    df_base = evaluate_pipeline(base_pipeline, image_dir, annotations_csv, file_list)
    score = df_base['total_score'].mean() if not df_base.empty else 0.0
    with open(BASELINE_FILE, 'w') as f:
        f.write(f"Baseline total_score: {score:.4f}")
        f.write(df_base.to_string(index=False))
    # Очищаем историю перед оптимизацией
    open(HISTORY_FILE, 'w').close()

# --------------------------------------
# Целевая функция для оптимизации
# --------------------------------------
def objective(phenotype):
    # собираем параметры
    dp = {}
    for i, key in enumerate(PARAM_BOUNDS):
        val = phenotype[i]
        dp[key] = int(val) if key == 'canvas_size' else val
    # применяем
    base_pipeline.detect_params = dp
    # оцениваем
    df = evaluate_pipeline(base_pipeline, image_dir, annotations_csv, file_list)
    score = df['total_score'].mean() if not df.empty else 0.0
    # логируем параметры и метрику в реальном времени
    with open(HISTORY_FILE, 'a') as hf:
        hf.write(f"Params: {dp}, Score: {score:.4f}\n")
    return score.mean() if not df.empty else 0.0

def ff(population):
    return np.array([objective(ind) for ind in population])

# --------------------------------------
# Запуск оптимизации
# --------------------------------------
if __name__ == '__main__':
    # сохраняем baseline до оптимизации
    save_baseline()
    left_border = np.array([bounds[0] for bounds in PARAM_BOUNDS.values()])
    left_border = np.array([bounds[0] for bounds in PARAM_BOUNDS.values()])
    right_border = np.array([bounds[1] for bounds in PARAM_BOUNDS.values()])
    num_vars = len(PARAM_BOUNDS)

    optimizer = DifferentialEvolution(
        fitness_function=ff,
        iters=20,
        pop_size=15,
        left_border=left_border,
        right_border=right_border,
        num_variables=num_vars,
        show_progress_each=1,
        minimization=False,
        mutation='current_to_best_1',
        F=0.5,
        CR=0.7,
        keep_history=True
    )
    optimizer.fit()

    fittest = optimizer.get_fittest()
    best_vector = fittest['phenotype']
    best_score = fittest['fitness']
    best_params = {}
    for i, key in enumerate(PARAM_BOUNDS):
        val = best_vector[i]
        best_params[key] = int(val) if key == 'canvas_size' else val

    print('Лучшие параметры:', best_params)
    print(f'Лучший средний total_score: {best_score:.4f}')