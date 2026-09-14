from ultralytics import YOLO

# To use YOLOv5, load a v5 weights file
# model = YOLO("yolov5su.pt")

# Migrate to the recommended YOLO26 model
model = YOLO("yolo26n.pt")

# Train on the COCO8 dataset using the efficient MuSGD optimizer
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=32,  # YOLO26's low memory footprint allows larger batch sizes
)

# Run an NMS-free inference
predictions = model("https://ultralytics.com/images/bus.jpg")
predictions[0].show()