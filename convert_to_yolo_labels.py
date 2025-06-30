import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# YOLO class order: K,Q,R,B,N,P,k,q,r,b,n,p
PIECE_CLASSES = ['K','Q','R','B','N','P','k','q','r','b','n','p']
PIECE_TO_CLASS = {p: i for i, p in enumerate(PIECE_CLASSES)}

DATASET_DIR = 'diverse_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels_yolo')
CSV_PATH = os.path.join(DATASET_DIR, 'labels.csv')

os.makedirs(LABELS_DIR, exist_ok=True)

def fen_to_piece_positions(fen):
    """Return list of (piece, rank, file) from FEN (rank 0=top, file 0=left)"""
    pieces = []
    ranks = fen.split(' ')[0].split('/')
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for c in rank:
            if c.isdigit():
                file_idx += int(c)
            elif c in PIECE_TO_CLASS:
                pieces.append((c, rank_idx, file_idx))
                file_idx += 1
    return pieces

def create_yolo_labels():
    df = pd.read_csv(CSV_PATH)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        label_path = os.path.join(LABELS_DIR, row['filename'].replace('.png', '.txt'))
        if not os.path.exists(img_path):
            continue
        # Open image to get size
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        square_w = w / 8
        square_h = h / 8
        yolo_lines = []
        for piece, rank, file in fen_to_piece_positions(row['fen']):
            class_id = PIECE_TO_CLASS[piece]
            # YOLO: x_center, y_center, width, height (normalized)
            x_center = (file + 0.5) * square_w / w
            y_center = (rank + 0.5) * square_h / h
            width = square_w / w
            height = square_h / h
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    print(f"✅ YOLO label files written to {LABELS_DIR}")

def write_data_yaml():
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(IMAGES_DIR)}\n")
        f.write(f"val: {os.path.abspath(IMAGES_DIR)}\n")
        f.write(f"nc: 12\n")
        f.write(f"names: {PIECE_CLASSES}\n")
    print(f"✅ data.yaml written to {yaml_path}")

if __name__ == "__main__":
    create_yolo_labels()
    write_data_yaml() 