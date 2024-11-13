import cv2
from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = model(frame, iou=0.4, conf=0.6)
    detect_frame = result[0].plot()

    cv2.imshow('YOLOv8-seg', detect_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
