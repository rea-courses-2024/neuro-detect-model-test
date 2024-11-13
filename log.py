import cv2
from ultralytics import YOLO
import logging
import os
import matplotlib.pyplot as plt

log_file = 'application.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def log_messages(message, level=logging.INFO):
    if level == logging.DEBUG:
        logging.debug(message)
    elif level == logging.INFO:
        logging.info(message)
    elif level == logging.WARNING:
        logging.warning(message)
    elif level == logging.ERROR:
        logging.error(message)
    elif level == logging.CRITICAL:
        logging.critical(message)


def main():
    model = YOLO('yolov8m-seg.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log_messages("Ошибка при открытии камеры", logging.ERROR)
        return

    log_messages("Камера успешно открыта", logging.INFO)

    while True:
        ret, frame = cap.read()
        if not ret:
            log_messages("Ошибка при чтении кадра", logging.ERROR)
            break

        result = model(frame, iou=0.4, conf=0.6)
        detect_frame = result[0].plot()

        cv2.imshow('YOLOv8-seg', detect_frame)

        log_messages("Кадр обработан", logging.DEBUG)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_messages("Выход из программы", logging.INFO)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()