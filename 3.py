import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Загрузка модели YOLOv8
model = YOLO('yolov8m-obb.pt')  # Замените на путь к вашей модели

# Функция для тестирования на датасете
def test_model_on_dataset(dataset_path):
    results = []

    # Перебор изображений в датасете
    for img_file in os.listdir(dataset_path):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(dataset_path, img_file)
            img = cv2.imread(img_path)

            # Выполнение детекции
            result = model(img)
            results.append((img_file, result))



    return results

# Составление отчета о результатах
# Составление отчета о результатах
def generate_report(results):
    report_path = 'test_report.txt'
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('Отчет о детектированных объектах:\n\n')

        for img_file, result in results:
            report_file.write(f'Результаты для изображения: {img_file}\n')
            report_file.write(f'Содержимое результата: {result}\n')

            obb = result[0].obb

            if obb is not None:
                report_file.write('Ориентированные ограничивающие рамки (OBB):\n')
                # Предполагаем, что obb имеет метод data для получения координат
                for ob in obb.data:
                    # Здесь предполагается, что ob содержит координаты OBB и класс
                    class_id = int(ob[-1])  # Предполагаем, что последний элемент - это класс
                    class_name = result[0].names[class_id]
                    report_file.write(f'Объект: {class_name}, Координаты OBB: {ob[:-1].tolist()}\n')  # Преобразуем в список для удобного чтения
            else:
                report_file.write('OBB не обнаружены.\n')

            report_file.write('\n')

    print(f'Отчет сгенерирован: {report_path}')

# Основной код
if __name__ == "__main__":
    datasets = ['images']  # Замените на пути к вашим датасетам
    all_results = []

    for dataset in datasets:
        print(f'Тестирование на датасете: {dataset}')
        results = test_model_on_dataset(dataset)
        all_results.extend(results)

    generate_report(all_results)
