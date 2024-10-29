import os

def convert_and_check_paths(input_file, base_path):
    # Создаем имя выходного файла, добавляя '_full_path' перед расширением
    file_root, file_ext = os.path.splitext(input_file)
    output_file = f"{file_root}_full_path{file_ext}"

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue  # пропускаем пустые строки
            parts = line.split(',', 1)  # разбиваем только по первому вхождению запятой
            if len(parts) != 2:
                print(f"Строка {line_num}: Пропущена из-за неверного формата: {line}")
                continue
            filename, label = parts
            filename = filename.strip()
            label = label.strip()
            # Создаем полный путь к файлу
            full_path = os.path.join(base_path, filename)
            # Проверяем, существует ли файл
            if os.path.exists(full_path):
                # Записываем полный путь и метку в выходной файл
                outfile.write(f"{full_path},{label}\n")
            else:
                print(f"Строка {line_num}: Файл не найден: {full_path}")

    print(f"Обработка файла {input_file} завершена. Результат сохранен в {output_file}.")

if __name__ == '__main__':
    base_path = r"C:\Users\user\Desktop\megacorpus\reports_u\img"

    # Список входных файлов
    input_files = [
        r"C:\Users\user\Desktop\megacorpus\reports_u\gt_reports_u_test.txt",
        r"C:\Users\user\Desktop\megacorpus\reports_u\gt_reports_u_training.txt"
    ]

    for input_file in input_files:
        convert_and_check_paths(input_file, base_path)
