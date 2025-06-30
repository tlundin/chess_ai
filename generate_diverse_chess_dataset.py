import os
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time

# Import the board generation functions from bitmap_chess_board.py
from bitmap_chess_board import (
    extract_piece_images, random_perspective, random_crop, 
    random_brightness_contrast, draw_board_with_pieces, find_perspective_coeffs
)

# Import PGN generator
from pgn_generator import PGNGenerator

# Configuration
BOARD_SIZE = 8
SQUARE_SIZE_RANGE = (60, 120)
CHESS_SET_PATH = 'chess_sets/1.png'
OUTPUT_DIR = 'diverse_dataset'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
CSV_PATH = os.path.join(OUTPUT_DIR, 'labels.csv')

# Add color combos at the top
COLOR_COMBOS = [
    ((240, 217, 181), (181, 136, 99)),    # a) #F0D9B5/#B58863
    ((237, 235, 210), (110, 109, 107)),   # b) #EDEBD2/#6E6D6B
    ((237, 235, 210), (113, 150, 80)),    # c) #EDEBD2/#719650
]

# Piece order in the image (left to right)
PIECE_ORDER = ['K', 'Q', 'B', 'N', 'R', 'P']

def generate_board_with_variations(fen, color_combo, board_id):
    piece_images = extract_piece_images(CHESS_SET_PATH)
    square_size = random.randint(*SQUARE_SIZE_RANGE)
    board_img = draw_board_with_pieces(fen, piece_images, square_size, None, colors=color_combo)
    augmentations_applied = []
    # Only brightness/contrast (80% chance)
    if random.random() < 0.8:
        board_img = random_brightness_contrast(board_img)
        augmentations_applied.append("brightness_contrast")
    # Only noise (20% chance)
    if random.random() < 0.2:
        img_array = np.array(board_img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255)
        board_img = Image.fromarray(img_array)
        augmentations_applied.append("noise")
    return board_img, augmentations_applied

def generate_single_board(args):
    board_id, pgn_gen, color_combo_idx = args
    fen = pgn_gen.generate_random_position()
    color_combo = COLOR_COMBOS[color_combo_idx]
    board_img, augmentations = generate_board_with_variations(fen, color_combo, board_id)
    image_filename = f"chess_board_{board_id:06d}.png"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    board_img.save(image_path)
    return {
        'filename': image_filename,
        'fen': fen,
        'chess_set': '1',
        'color_combo': color_combo_idx,
        'augmentations': ','.join(augmentations) if augmentations else 'none',
        'split': 'train' if board_id < int(75000 * 0.8) else 'val'
    }

def create_dataset_parallel(num_samples_per_combo=25000):
    print(f"🎯 Generating {num_samples_per_combo*3} chess board images using parallel processing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    num_cores = mp.cpu_count()
    print(f"🖥️  Using {num_cores} CPU cores for parallel processing")
    pgn_gens = [PGNGenerator() for _ in range(num_cores)]
    args = []
    for color_combo_idx in range(3):
        for i in range(num_samples_per_combo):
            board_id = color_combo_idx * num_samples_per_combo + i
            args.append((board_id, pgn_gens[board_id % num_cores], color_combo_idx))
    batch_size = 1000
    all_data = []
    start_time = time.time()
    total_samples = num_samples_per_combo * 3
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_args = args[batch_start:batch_end]
        print(f"📊 Processing batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} "
              f"({batch_start}-{batch_end-1})")
        with mp.Pool(processes=num_cores) as pool:
            batch_data = pool.map(generate_single_board, batch_args)
        all_data.extend(batch_data)
        elapsed = time.time() - start_time
        rate = (batch_end) / elapsed
        eta = (total_samples - batch_end) / rate if rate > 0 else 0
        print(f"   ✅ Batch complete. Rate: {rate:.1f} boards/sec, ETA: {eta/60:.1f} minutes")
    df = pd.DataFrame(all_data)
    df.to_csv(CSV_PATH, index=False)
    total_time = time.time() - start_time
    print(f"\n✅ Dataset created successfully!")
    print(f"📁 Images saved to: {IMAGES_DIR}")
    print(f"📄 Labels saved to: {CSV_PATH}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Training samples: {len(df[df['split'] == 'train'])}")
    print(f"📊 Validation samples: {len(df[df['split'] == 'val'])}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {total_samples/total_time:.1f} boards/sec")
    print(f"\n📈 Dataset Statistics:")
    print(f"   Color combos used: {df['color_combo'].value_counts().to_dict()}")
    print(f"   Augmentations applied: {df['augmentations'].value_counts().head(10).to_dict()}")
    return df

def create_dataset_sequential(num_samples=50000):
    """Create the complete dataset using sequential processing (fallback)"""
    print(f"🎯 Generating {num_samples} chess board images (sequential mode)...")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Create PGN generator
    pgn_gen = PGNGenerator()
    
    # Prepare data for CSV
    data = []
    
    start_time = time.time()
    
    for i in range(num_samples):
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (num_samples - i) / rate if rate > 0 else 0
            print(f"📊 Progress: {i}/{num_samples} ({i/num_samples*100:.1f}%) - "
                  f"Rate: {rate:.1f} boards/sec, ETA: {eta/60:.1f} minutes")
        
        # Generate random FEN using PGN generator
        fen = pgn_gen.generate_random_position()
        
        # Randomly select chess set
        chess_set_path = random.choice(CHESS_SET_PATH)
        chess_set_id = chess_set_path.split('/')[-1].split('.')[0]
        
        # Generate board with variations
        board_img, augmentations = generate_board_with_variations(fen, CHESS_SET_PATH, i)
        
        # Save image
        image_filename = f"chess_board_{i:06d}.png"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        board_img.save(image_path)
        
        # Prepare data for CSV
        data.append({
            'filename': image_filename,
            'fen': fen,
            'chess_set': chess_set_id,
            'augmentations': ','.join(augmentations) if augmentations else 'none',
            'split': 'train' if i < int(num_samples * 0.8) else 'val'
        })
    
    # Create and save CSV
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    
    total_time = time.time() - start_time
    
    print(f"✅ Dataset created successfully!")
    print(f"📁 Images saved to: {IMAGES_DIR}")
    print(f"📄 Labels saved to: {CSV_PATH}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Training samples: {len(df[df['split'] == 'train'])}")
    print(f"📊 Validation samples: {len(df[df['split'] == 'val'])}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {num_samples/total_time:.1f} boards/sec")
    
    # Print some statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Chess sets used: {df['chess_set'].value_counts().to_dict()}")
    print(f"   Augmentations applied: {df['augmentations'].value_counts().head(10).to_dict()}")
    
    return df

def main():
    print("🚀 Starting Diverse Chess Dataset Generation (3 color combos, 75,000 boards)")
    print("=" * 50)
    if not os.path.exists(CHESS_SET_PATH):
        print(f"❌ Chess set not found: {CHESS_SET_PATH}")
        return
    print("✅ Chess set found!")
    try:
        print("🔄 Attempting parallel processing...")
        df = create_dataset_parallel(25000)
    except Exception as e:
        print(f"⚠️  Parallel processing failed: {e}")
        print("🔄 Falling back to sequential processing... (not implemented for 3 color combos)")
    print(f"\n🎉 Dataset ready for training!")

if __name__ == "__main__":
    main() 