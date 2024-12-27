import sys
sys.path.insert(0, 'ultralytics/')

import ultralytics
print(ultralytics.checks())

from ultralytics import YOLO

# Load a model
model = YOLO('yolo11-cus.yaml', task='detect')

# Train the model
model.train(data="coco.yaml", epochs=2, imgsz=640, device='0,1,2,3', batch=128)