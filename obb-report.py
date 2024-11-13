import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('yolov8l-obb.pt')


def test_model_on_dataset(dataset_path):
    results = []

    for img_file in os.listdir(dataset_path):
        if img_file.endswith(('.jpeg', '.png', '.bmp', '.jpg')):
            img_path = os.path.join(dataset_path, img_file)
            img = cv2.imread(img_path)

            result = model(img)
            results.append((img_file, result))

    return results


def generated_report(results):
    report_path = 'obb-report.txt'
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('Отчет о работе модели yolov8m-obb\n')

        for img_file, result in results:
            report_file.write(f'Результаты для изображения: {img_file} \n')
            report_file.write(f'Результат: {result}\n')

            obb = result[0].obb

            if obb is not None:
                report_file.write('Найденные объекты типа OBB:')

                for ob in obb.data:
                    class_id = int(ob[-1])
                    class_name = result[0].names[class_id]
                    report_file.write(f'Объект: {class_name},'
                                      f' Координаты: {ob[:-1].tolist()}\n')

                else:
                    report_file.write('Такие объекты не обнаружены \n')

                report_file.write('\n')


if __name__ == "__main__":
    datasets = ['images']
    all_result = []

    for dataset in datasets:
        results = test_model_on_dataset(dataset)
        all_result.extend(results)

    generated_report(all_result)
