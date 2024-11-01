import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Путь к папке с изображениями
folder_path = r"C:\Users\user\Desktop\megacorpus\reports_u\img"  # Замените на путь к вашей папке с изображениями

# Списки для хранения высот и ширин
heights = []
widths = []

# Чтение изображений и получение их размеров
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            h, w, _ = image.shape
            heights.append(h)
            widths.append(w)

# Вычисление квартилей для высоты и ширины
height_quartiles = np.percentile(heights, [25, 50, 75])
width_quartiles = np.percentile(widths, [25, 50, 75])

print("Квартильные значения для высоты:", height_quartiles)
print("Квартильные значения для ширины:", width_quartiles)

# Построение ящиков с усами для высоты и ширины
plt.figure(figsize=(8, 10))  # Устанавливаем вертикальную фигуру
plt.style.use('seaborn-whitegrid')  # Устанавливаем стиль

# Определение цветов для боксплотов
colors = ['#3498db', '#e74c3c']

# Ящик с усами для высоты
plt.subplot(2, 1, 1)
plt.boxplot(heights, vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor=colors[0], color=colors[0]),
            whiskerprops=dict(color=colors[0]),
            capprops=dict(color=colors[0]),
            medianprops=dict(color='black'),
            flierprops=dict(markerfacecolor=colors[0], marker='o', markersize=5))
plt.title("Boxplot высоты изображений", fontsize=14, fontweight='bold')
plt.ylabel("Высота (пиксели)")
plt.xticks([1], ['Высота'])  # Устанавливаем название на оси x

# Ящик с усами для ширины
plt.subplot(2, 1, 2)
plt.boxplot(widths, vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor=colors[1], color=colors[1]),
            whiskerprops=dict(color=colors[1]),
            capprops=dict(color=colors[1]),
            medianprops=dict(color='black'),
            flierprops=dict(markerfacecolor=colors[1], marker='o', markersize=5))
plt.title("Boxplot ширины изображений", fontsize=14, fontweight='bold')
plt.ylabel("Ширина (пиксели)")
plt.xticks([1], ['Ширина'])  # Устанавливаем название на оси x

plt.tight_layout()
plt.show()
