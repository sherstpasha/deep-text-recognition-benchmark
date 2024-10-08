import re

# Указываем путь к исходному файлу и путь к файлу с отфильтрованными результатами
input_file = r"C:\Users\user\Desktop\megacorpus\gt_test.txt"
output_file = r"C:\Users\user\Desktop\megacorpus\different_gt\reports_gt_test.txt"

# Указываем папки, которые нужно оставить
include_folders = ["reports"]

# Регулярное выражение для поиска конкретных папок в пути
pattern = re.compile(r"\\([^\\]+)\\")  # Извлечение имен папок из пути

# Указываем исходный и новый корневой путь, который нужно заменить
old_root_path = r"C:\Users\user\Desktop\megacorpus"
new_root_path = "mounted_folder"

# Считываем исходный файл
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Фильтруем строки и заменяем часть пути
filtered_lines = []
for line in lines:
    # Извлекаем все папки из пути
    match_folders = pattern.findall(line)
    # Проверяем, есть ли хотя бы одна папка из списка в извлеченных папках
    if any(folder in include_folders for folder in match_folders):
        # Заменяем старый корневой путь на новый
        modified_line = line.replace(old_root_path, new_root_path)
        # Меняем все обратные слеши (\) на прямые (/)
        modified_line = modified_line.replace("\\", "/")
        filtered_lines.append(modified_line)

# Сохраняем отфильтрованные строки в новый файл
with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(filtered_lines)

print(f"Отфильтрованные и измененные пути сохранены в файл: {output_file}")
