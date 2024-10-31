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
plt.figure(figsize=(10, 5))

# Ящик с усами для высоты
plt.subplot(1, 2, 1)
plt.boxplot(heights, vert=True, patch_artist=True)
plt.title("Квартильное распределение высоты изображений")
plt.ylabel("Высота (пиксели)")

# Ящик с усами для ширины
plt.subplot(1, 2, 2)
plt.boxplot(widths, vert=True, patch_artist=True)
plt.title("Квартильное распределение ширины изображений")
plt.ylabel("Ширина (пиксели)")

plt.tight_layout()
plt.show()
