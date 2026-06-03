from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    "test_images",
    conf=0.25,
    save=True
)

print("Predictions completed.")