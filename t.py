import subprocess
import os
import time

max_parallel_processes = 1  # Максимальное количество параллельных процессов
processes = []  # Список для хранения запущенных процессов

for i in range(1, 9):
    config_file = f'config_{i}.yaml'
    if not os.path.exists(config_file):
        print(f"Файл конфигурации {config_file} не найден. Пропускаем.")
        continue
    command = ['python', 'train_from_config.py', '--config', config_file]
    print(f"Запускаем: {' '.join(command)}")
    # Запускаем процесс
    process = subprocess.Popen(command)
    processes.append(process)
    # Если достигли максимального количества параллельных процессов, ждем завершения одного из них
    while len(processes) >= max_parallel_processes:
        # Проверяем, завершился ли какой-либо процесс
        for p in processes:
            if p.poll() is not None:  # Процесс завершился
                processes.remove(p)
                break
        else:
            # Если ни один процесс еще не завершился, ждем немного
            time.sleep(1)

# Ждем завершения оставшихся процессов
for p in processes:
    p.wait()
