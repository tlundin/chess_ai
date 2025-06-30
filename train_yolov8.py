from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8l.pt')  # Load a pretrained YOLOv8-L model

    # Train the model
    model.train(
        data='diverse_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        optimizer='AdamW',
        batch=32,  # or your preferred batch size
        project='yolo_train',
        name='yolo8l-diverse',
        device=0  # Use GPU 0
    )