import os
import uuid
import random
import csv
from datetime import datetime
from pgn_generator import PGNGenerator
from PIL import Image, ImageDraw, PngImagePlugin

# Settings
NUM_IMAGES = 10000
OUTPUT_DIR = 'dataset'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
LABELS_CSV = os.path.join(OUTPUT_DIR, 'labels.csv')
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# Board and piece settings
BOARD_SIZE = 8
PIECE_IMAGE = 'chess_pieces.png'
PIECE_ORDER = ['K', 'Q', 'B', 'N', 'R', 'P']
PIECE_TO_INDEX = {
    'K': (0, 0), 'Q': (0, 1), 'B': (0, 2), 'N': (0, 3), 'R': (0, 4), 'P': (0, 5),
    'k': (1, 0), 'q': (1, 1), 'b': (1, 2), 'n': (1, 3), 'r': (1, 4), 'p': (1, 5),
}

def extract_piece_images(sheet_path):
    sheet = Image.open(sheet_path).convert('RGBA')
    w, h = sheet.size
    piece_w = w // 6
    piece_h = h // 2
    pieces = {}
    for row in range(2):
        for col in range(6):
            box = (col * piece_w, row * piece_h, (col + 1) * piece_w, (row + 1) * piece_h)
            piece_img = sheet.crop(box)
            letter = PIECE_ORDER[col]
            if row == 0:
                pieces[letter] = piece_img
            else:
                pieces[letter.lower()] = piece_img
    return pieces, piece_w

def draw_board_with_pieces(fen, piece_images, square_size, out_path, pgn_text=None):
    board_img = Image.new('RGBA', (BOARD_SIZE * square_size, BOARD_SIZE * square_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(board_img)
    colors = [(240, 217, 181), (181, 136, 99)]
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            color = colors[(rank + file) % 2]
            x1 = file * square_size
            y1 = rank * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            draw.rectangle([x1, y1, x2, y2], fill=color)
    fen_ranks = fen.split('/')
    for rank_idx, fen_rank in enumerate(fen_ranks):
        file_idx = 0
        for char in fen_rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                piece_img = piece_images.get(char)
                if piece_img:
                    x = file_idx * square_size
                    y = rank_idx * square_size
                    board_img.paste(piece_img.resize((square_size, square_size), Image.Resampling.LANCZOS), (x, y), piece_img)
                file_idx += 1
    if pgn_text is not None:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pgn", pgn_text)
        board_img.save(out_path, pnginfo=meta)
    else:
        board_img.save(out_path)

def split_indices(num_items, ratios):
    indices = list(range(num_items))
    random.shuffle(indices)
    n_train = int(num_items * ratios['train'])
    n_val = int(num_items * ratios['val'])
    n_test = num_items - n_train - n_val
    return (
        set(indices[:n_train]),
        set(indices[n_train:n_train+n_val]),
        set(indices[n_train+n_val:])
    )

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    pgn_gen = PGNGenerator()
    piece_images, piece_size = extract_piece_images(PIECE_IMAGE)
    records = []
    print(f"Generating {NUM_IMAGES} chessboard images for ML dataset...")
    for i in range(NUM_IMAGES):
        fen = pgn_gen.generate_random_position()
        img_id = str(uuid.uuid4())
        filename = f"chess_board_{img_id}.png"
        img_path = os.path.join(IMAGES_DIR, filename)
        draw_board_with_pieces(fen, piece_images, piece_size, img_path, pgn_text=fen)
        records.append({'filename': filename, 'pgn': fen})
        print(f"  {i+1}/{NUM_IMAGES}: {filename}")
    # Split into train/val/test
    n = len(records)
    train_idx, val_idx, test_idx = split_indices(n, SPLIT_RATIOS)
    for i, rec in enumerate(records):
        if i in train_idx:
            rec['split'] = 'train'
        elif i in val_idx:
            rec['split'] = 'val'
        else:
            rec['split'] = 'test'
    # Write CSV
    with open(LABELS_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'pgn', 'split'])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"\nDataset complete! Images in {IMAGES_DIR}, labels in {LABELS_CSV}")

if __name__ == '__main__':
    main() 