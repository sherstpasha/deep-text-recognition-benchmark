from train_func import train

train(
    train_csv=r"/workspace/mounted_folder/labels.csv",  # путь до сsv файла с расшифровками для обучения
    train_root=r"/workspace/mounted_folder/img",  # путь до папки с изображениями для обучения
    valid_csv=r"/workspace/mounted_folder/labels.csv",  # путь до сsv файла с расшифровками для валидации
    valid_root=r"/workspace/mounted_folder/img",  # путь до папки с изображениями для валидации
    batch_size=16,  # размер батча
    num_iter=100,  # количество итераций
    valInterval=10,  # интервал валидации
    saved_model="",  # путь до модели, от которой начинать обучение. Если пусто, то обучение с нуля
    output_dir="test",  # путь до директории, в которой сохраняется результат
)
