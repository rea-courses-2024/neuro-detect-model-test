import os
import unittest
import cv2
from ultralytics import YOLO


class TestYOLOReportObject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = YOLO('yolov8m.pt')

    def test_report_generation(self):
        results = []
        dataset_path = 'images'
        dog_detection = False

        for img_file in os.listdir(dataset_path):
            if img_file.endswith(('.jpeg', '.png', '.bmp', '.jpg')):
                img_path = os.path.join(dataset_path, img_file)
                img = cv2.imread(img_path)
                result = self.model(img)
                results.append((img_file, result))

        report_path = 'report_detect_object.txt'
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('Отчет о работе модели yolov8m\n')

            for img_file, result in results:
                report_file.write(f'Результаты для изображения: {img_file} \n')
                self.assertIsNotNone(result, 'Результат должен быть не None\n')

                boxes = result[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes.data:
                        self.assertIsNotNone(
                            box, 'Результат должен быть не None\n')

                        x1, y1, x2, y2, conf, class_id = box.tolist()
                        class_name = self.model.names[int(class_id)]
                        report_file.write(
                            f"Обнаружен объект класса {class_name},"
                            f" уверенность {conf}\n")

                        if class_name == 'dog':
                            dog_detection = True

                else:
                    report_file.write('Объекты не обнаружены \n')

            self.assertTrue(dog_detection,
                            "Ожидаемый объект класса dog найден не был")

            report_file.write('Тест пройден: объект найден')


if __name__ == '__main__':
    unittest.main()
