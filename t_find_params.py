import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from hardocr import DocumentOCRPipeline
from thefittest.optimizers import DifferentialEvolution

# --------------------------------------
# Константы и параметры оптимизации
# --------------------------------------
PARAM_BOUNDS = {
    'text_threshold': (0.1, 1.0),
    'low_text': (0.0, 0.6),
    'link_threshold': (0.0, 1.0),
    'canvas_size': (256, 2560),
    'mag_ratio': (0.5, 3.0),
    'slope_ths': (0.0, 1.0),
    'ycenter_ths': (0.0, 1.0),
    'height_ths': (0.0, 1.0),
    'width_ths': (0.0, 1.0),
    'add_margin': (0.0, 0.5),
    'contrast': (0.0, 0.5),
    'sharpness': (0.0, 0.5),
    'brightness': (0.0, 0.5),
    'gamma': (0.8, 1.2),
}

BEST_DETECT_PARAMS = {
    'text_threshold': 0.5213,
    'low_text': 0.3616,
    'link_threshold': 0.3211,
    'canvas_size': 2560,
    'mag_ratio': 2.4571,
    'slope_ths': 0.4065,
    'ycenter_ths': 0.0,
    'height_ths': 0.7308,
    'width_ths': 0.6589,
    'add_margin': 0.0721
}

IMAGE_DIR = r"C:\\shared\\Archive_19_04\\combined_images"
ANNOTATIONS_CSV = r"C:\\shared\\Archive_19_04\\annotations_with_image_size_c.csv"
CONFIG_PATH = 'config.yaml'
OCR_MODEL_PATH = r"C:\\shared\\saved_models_25_04\\TPS-ResNet-BiLSTM-Attn-Seed1111\\best_norm_ED.pth"
HISTORY_FILE = 'de_history.txt'
BASELINE_FILE = 'baseline_stats.txt'

# --------------------------------------
# Инициализация пайплайна
# --------------------------------------

def create_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = DocumentOCRPipeline(
        config_path=CONFIG_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        device=device,
        detect_params=BEST_DETECT_PARAMS.copy(),
    )
    pipeline.contrast = 0.0
    pipeline.sharpness = 0.0
    pipeline.brightness = 0.0
    pipeline.gamma = 1.0
    return pipeline

base_pipeline = create_pipeline()

# --------------------------------------
# Метрики
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
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    dp[:, 0] = np.arange(len1 + 1)
    dp[0, :] = np.arange(len2 + 1)
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return dp[len1, len2]

# --------------------------------------
# Оценка OCR пайплайна
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
        truths = [((row['x_center']*iw - row['box_width']*iw/2,
                    row['y_center']*ih - row['box_height']*ih/2,
                    row['x_center']*iw + row['box_width']*iw/2,
                    row['y_center']*ih + row['box_height']*ih/2), str(row['decoding']))
                  for _, row in ann.iterrows()]
        
        ious, dices, norm_eds = [], [], []
        correct, total = 0, 0
        used = set()

        for t_box, t_text in truths:
            best_iou, best_idx, best_pt = 0.0, None, ''
            for idx, (p_box, p_text) in enumerate(preds):
                if idx in used: continue
                iou = compute_iou(t_box, p_box)
                if iou > best_iou:
                    best_iou, best_idx, best_pt = iou, idx, p_text
            ious.append(best_iou)
            if best_iou >= iou_thresh and best_idx is not None:
                used.add(best_idx)
                dices.append(compute_dice(t_box, preds[best_idx][0]))
                total += 1
                correct += (best_pt == t_text)
                norm_eds.append(1 - edit_distance(best_pt, t_text) / max(len(t_text), 1))
            else:
                dices.append(0.0)
                norm_eds.append(0.0)

        if not ious: continue
        results.append({
            'image': image_name,
            'mean_iou': np.mean(ious),
            'mean_dice': np.mean(dices),
            'recog_accuracy': correct/total if total else 0.0,
            'mean_norm_score': np.mean(norm_eds),
            'num_true': len(truths),
            'num_pred': len(preds),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df['total_score'] = df['mean_iou'] + df['recog_accuracy'] + df['mean_norm_score']
    return df

# --------------------------------------
# Целевая функция для оптимизации
# --------------------------------------

def objective(phenotype):
    params = dict(zip(PARAM_BOUNDS.keys(), phenotype))

    detect_params = {k: int(v) if k == 'canvas_size' else v
                     for k, v in params.items() if k not in ('contrast', 'sharpness', 'brightness', 'gamma')}
    base_pipeline.detect_params = detect_params
    base_pipeline.contrast = params['contrast']
    base_pipeline.sharpness = params['sharpness']
    base_pipeline.brightness = params['brightness']
    base_pipeline.gamma = params['gamma']

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')][::100]
    df = evaluate_pipeline(base_pipeline, IMAGE_DIR, ANNOTATIONS_CSV, files)
    score = df['total_score'].mean() if not df.empty else 0.0

    with open(HISTORY_FILE, 'a') as f:
        f.write(f"Params: {params}, Score: {score:.4f}\n")

    return score

# --------------------------------------
# Главный запуск
# --------------------------------------

if __name__ == '__main__':
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')][::100]
    df_base = evaluate_pipeline(base_pipeline, IMAGE_DIR, ANNOTATIONS_CSV, files)
    base_score = df_base['total_score'].mean() if not df_base.empty else 0.0

    with open(BASELINE_FILE, 'w') as f:
        f.write(f"Baseline total_score: {base_score:.4f}\n")

    open(HISTORY_FILE, 'w').close()

    left = np.array([v[0] for v in PARAM_BOUNDS.values()])
    right = np.array([v[1] for v in PARAM_BOUNDS.values()])
    num_vars = len(PARAM_BOUNDS)

    optimizer = DifferentialEvolution(
        fitness_function=lambda pop: np.array([objective(ind) for ind in pop]),
        iters=20,
        pop_size=15,
        left_border=left,
        right_border=right,
        num_variables=num_vars,
        show_progress_each=1,
        minimization=False,
        mutation='current_to_best_1',
        F=0.5,
        CR=0.7,
        keep_history=True
    )
    optimizer.fit()

    best = optimizer.get_fittest()
    print('Лучшие параметры:', dict(zip(PARAM_BOUNDS.keys(), best['phenotype'])))
    print('Лучший score:', best['fitness'])