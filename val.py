import sys
sys.path.insert(0, './ultralytics')

import ultralytics
print(ultralytics.checks())

from ultralytics import YOLO

# Load a model
model = YOLO('runs/train/weights/best.pt', task='detect')

# Val the model
model.val(data="coco.yaml", batch=128, device="0,1,2,3", split='test', project='test')
