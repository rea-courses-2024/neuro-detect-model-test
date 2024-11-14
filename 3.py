from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data="YOLO_dataset/data.yaml", epochs=5, iterations=5, optimizer="AdamW", plots=False, save=False, val=False)