import random

def split_data(file_path):
    # Чтение содержимого исходного файла
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Перемешивание строк случайным образом
    random.shuffle(lines)

    # Разделение данных на тренировочную и тестовую части
    split_index = int(len(lines) * 0.77)
    training_data = lines[:split_index]
    test_data = lines[split_index:]

    # Сохранение результатов в новые файлы
    training_file_path = file_path.replace(".txt", "_training.txt")
    test_file_path = file_path.replace(".txt", "_test.txt")

    with open(training_file_path, "w", encoding="utf-8") as train_file:
        train_file.writelines(training_data)

    with open(test_file_path, "w", encoding="utf-8") as test_file:
        test_file.writelines(test_data)

    print(f"Тренировочные данные сохранены в файл: {training_file_path}")
    print(f"Тестовые данные сохранены в файл: {test_file_path}")

# Пример вызова функции:
split_data(r"C:\Users\user\Desktop\megacorpus\reports\gt_reports.txt")
