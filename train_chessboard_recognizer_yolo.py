from ultralytics import YOLO
import torch
import os

# Paths
DATA_YAML = 'diverse_dataset/data.yaml'
OUTPUT_DIR = 'yolo_training_output'
MODEL_NAME = 'yolov8l.pt'  # COCO-pretrained YOLOv8-L
EPOCHS = 100
IMG_SIZE = 640
BATCH = 0  # 0 = auto
OPTIMIZER = 'AdamW'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model = YOLO(MODEL_NAME)

# Train
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    optimizer=OPTIMIZER,
    device=device,
    project=OUTPUT_DIR,
    name='chessboard_yolo',
    pretrained=True,
    verbose=True,
    exist_ok=True,
    save=True,
    save_period=1,
    patience=20,
    lr0=0.001,
    close_mosaic=0,  # keep strong augmentations
)

# Save best model to root
best_model_path = os.path.join(OUTPUT_DIR, 'chessboard_yolo', 'weights', 'best.pt')
if os.path.exists(best_model_path):
    os.rename(best_model_path, 'best_chessboard_yolo.pt')
    print(f"✅ Best model saved as best_chessboard_yolo.pt")
else:
    print(f"❌ Best model not found at {best_model_path}")

# Print validation metrics
print("\n=== Validation Results ===")
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH, device=device)
print(metrics) 