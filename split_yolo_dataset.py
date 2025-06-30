import os
import random
import shutil
from glob import glob

# Parameters
IMAGE_DIR = 'diverse_dataset/images'
LABEL_DIR = 'diverse_dataset/labels_yolo'
NEW_IMAGE_DIR = 'diverse_dataset/images'
NEW_LABEL_DIR = 'diverse_dataset/labels'
TRAIN_RATIO = 0.8

# Create new directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(NEW_IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(NEW_LABEL_DIR, split), exist_ok=True)

# Get all image files
image_files = glob(os.path.join(IMAGE_DIR, '*.jpg')) + glob(os.path.join(IMAGE_DIR, '*.png'))
image_files.sort()
random.shuffle(image_files)

# Split
split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Move files
def move_files(files, split):
    for img_path in files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base + '.txt')
        # Move image
        shutil.move(img_path, os.path.join(NEW_IMAGE_DIR, split, os.path.basename(img_path)))
        # Move label if exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(NEW_LABEL_DIR, split, os.path.basename(label_path)))
        else:
            print(f'Warning: No label for {img_path}')

move_files(train_files, 'train')
move_files(val_files, 'val')

print(f'Total images: {len(image_files)}')
print(f'Train: {len(train_files)}')
print(f'Val: {len(val_files)}')
print('Done!') 