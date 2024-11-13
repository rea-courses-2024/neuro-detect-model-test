import unittest
import cv2
import numpy as np
from ultralytics import YOLO


class TestYOLOv8Segmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = YOLO('yolov8m-seg.pt')

    def test_model_loading(self):
        self.assertIsNotNone(self.model, "Модель не должна быть None")

    def test_frame_reading(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Кадр не был корректно считан")
        self.assertIsInstance(frame, (np.ndarray,),
                              "Кадр должен быть массивом numpy")

    def test_detection_output(self):
        frame = cv2.imread("test_image.jpg")
        result = self.model(frame, iou=0.4, conf=0.6)
        self.assertGreater(len(result[0].boxes), 0,
                           "Обнаруженных объектов должно быть больше 0")

    def test_detection_plot(self):
        frame = cv2.imread("test_image.jpg")
        result = self.model(frame, iou=0.4, conf=0.6)
        detect_frame = result[0].plot()
        self.assertIsInstance(detect_frame, np.ndarray,
                              "Результат визуализации не массив numpy"
                              " (не корректен)")

    def test_video_capture_release(self):
        cap = cv2.VideoCapture(0)
        self.assertTrue(cap.isOpened(),
                        "Камера не должна быть закрыта")
        cap.release()
        self.assertFalse(cap.isOpened(),
                         "Камера должна быть закрыта после release")


if __name__ == "__main__":
    unittest.main()
