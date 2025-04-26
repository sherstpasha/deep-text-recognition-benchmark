import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from hardocr import DocumentOCRPipeline
from matplotlib import pyplot as plt

# --------------------------------------
# Функции метрик
# --------------------------------------
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

def compute_dice(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return 2 * interArea / float(areaA + areaB)

def edit_distance(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0]*(len2+1) for _ in range(len1+1)]
    for i in range(len1+1): dp[i][0] = i
    for j in range(len2+1): dp[0][j] = j
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len1][len2]

# --------------------------------------
# Оценка пайплайна
# --------------------------------------
def evaluate_pipeline(pipeline, image_dir, annotations_csv, file_list, iou_thresh=0.5):
    df_ann = pd.read_csv(annotations_csv)
    totals = []
    for image_name in tqdm(file_list, desc="Evaluating on subset"):  # progress bar
        path = os.path.join(image_dir, image_name)
        if not os.path.exists(path): continue
        img = Image.open(path)
        page = pipeline(img)
        preds = [((w.x1, w.y1, w.x2, w.y2), w.text)
                 for blk in page.blocks for strn in blk.strings for w in strn.words]
        ann = df_ann[df_ann['image_name']==image_name]
        if ann.empty: continue
        iw, ih = ann['image_width'].iloc[0], ann['image_height'].iloc[0]
        truths = []
        for _, row in ann.iterrows():
            xc, yc = row['x_center']*iw, row['y_center']*ih
            bw, bh = row['box_width']*iw, row['box_height']*ih
            x1, y1 = xc-bw/2, yc-bh/2
            x2, y2 = xc+bw/2, yc+bh/2
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
                norm_eds.append(1 - ed/max(len(t_text),1))
            else:
                dices.append(0.0)
                norm_eds.append(0.0)
        if not ious: continue
        mean_iou = np.mean(ious)
        recog_acc = correct/total if total>0 else 0.0
        mean_score = mean_iou + recog_acc + np.mean(norm_eds)
        totals.append(mean_score)
    return np.mean(totals) if totals else 0.0

# --------------------------------------
# Извлечение лучших параметров из истории
# --------------------------------------
def load_best_params(history_file):
    best = []
    max_score = -np.inf
    pattern = re.compile(r"Params: (\{.*\}), Score: ([0-9.]+)")
    with open(history_file) as f:
        for line in f:
            m = pattern.search(line)
            if not m: continue
            params = eval(m.group(1))
            score = float(m.group(2))
            if score > max_score:
                best.append(params)
                max_score = score
    return best

# --------------------------------------
# Основной блок
# --------------------------------------
if __name__ == '__main__':
    image_dir = r"C:\shared\Archive_19_04\combined_images"
    annotations_csv = r"C:\shared\Archive_19_04\annotations_with_image_size_c.csv"
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    train_list = all_files[::100]
    test_list = all_files[::99]

    # baseline
    DEFAULT_PARAMS = {
    }
    
    pipeline = DocumentOCRPipeline(config_path='config.yaml',
                                   ocr_model_path = r"C:\shared\saved_models_25_04\TPS-ResNet-BiLSTM-Attn-Seed1111\best_norm_ED.pth",
                                   device=torch.device('cuda'),
                                   detect_params=DEFAULT_PARAMS)
    baseline_score = evaluate_pipeline(pipeline, image_dir, annotations_csv, test_list)

    # загрузка лучших параметров
    best_params_list = load_best_params('de_history.txt')

    # оценки на train и test
    train_scores = []
    test_scores = []
    for params in best_params_list:
        pipeline.detect_params = params
        train_scores.append(evaluate_pipeline(pipeline, image_dir, annotations_csv, train_list))
        test_scores.append(evaluate_pipeline(pipeline, image_dir, annotations_csv, test_list))

    # построение графика
    plt.figure(figsize=(10,6))
    x = range(1, len(best_params_list)+1)
    plt.plot(x, train_scores, marker='o', label='Train avg total_score')
    plt.plot(x, test_scores, marker='s', label='Test avg total_score')
    plt.axhline(baseline_score, color='gray', linestyle='--', label='Baseline (test)')
    plt.xlabel('Iteration of best improvement')
    plt.ylabel('Average total_score')
    plt.legend()
    plt.title('Performance over best parameter snapshots')
    plt.tight_layout()
    plt.show()
