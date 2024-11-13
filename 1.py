import os
import unittest
import cv2
from ultralytics import YOLO


class TestYOLOReportGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = YOLO('yolov8m.pt')

    def test_report_generation(self):
        results = []
        dataset_path = 'images'  # Путь к вашему датасету

        for img_file in os.listdir(dataset_path):
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(dataset_path, img_file)
                img = cv2.imread(img_path)
                result = self.model(img)
                results.append((img_file, result))

        report_path = 'test_report.txt'
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('Отчет о детектированных объектах:\n\n')
            for img_file, result in results:
                report_file.write(f'Результаты для изображения: {img_file}\n')
                self.assertIsNotNone(result, "Результат должен быть не None")

                # Проверка наличия коробок
                boxes = result[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes.data:
                        self.assertIsNotNone(box, "Коробка должна быть не None")
                        # Извлечение координат и класса
                        x1, y1, x2, y2, confidence, class_id = box.tolist()
                        class_name = self.model.names[int(class_id)]  # Получение имени класса
                        report_file.write(
                            f'Обнаружен объект: {class_name}, координаты: [{x1}, {y1}, {x2}, {y2}], уверенность: {confidence}\n')
                else:
                    report_file.write('Объекты не обнаружены.\n')


if __name__ == '__main__':
    unittest.main()
