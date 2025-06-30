print('Script started')
import random
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import os

# Board and piece settings
BOARD_SIZE = 8
SQUARE_SIZE_RANGE = (60, 120)  # pixels, min and max
CHESS_SET_PATHS = [f'chess_sets/{i}.png' for i in [1,2,3]]
OUTPUT_DIR = 'generated_boards'
OUTPUT_IMAGE_TEMPLATE = os.path.join(OUTPUT_DIR, 'bitmap_chess_board_{}.png')

# FEN for starting position
START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

# Piece order in the image (left to right)
PIECE_ORDER = ['K', 'Q', 'B', 'N', 'R', 'P']

# Map FEN letters to piece image index (row, col)
PIECE_TO_INDEX = {
    'K': (0, 0), 'Q': (0, 1), 'B': (0, 2), 'N': (0, 3), 'R': (0, 4), 'P': (0, 5),
    'k': (1, 0), 'q': (1, 1), 'b': (1, 2), 'n': (1, 3), 'r': (1, 4), 'p': (1, 5),
}

def extract_piece_images(sheet_path):
    print(f'Extracting pieces from {sheet_path}')
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
    print('Piece extraction complete')
    return pieces

def random_perspective(img):
    print('Applying random perspective')
    width, height = img.size
    margin = int(0.1 * min(width, height))
    def rand_pt(x, y):
        return (
            x + random.randint(-margin, margin),
            y + random.randint(-margin, margin)
        )
    src = [(0,0), (width,0), (width,height), (0,height)]
    dst = [rand_pt(x, y) for (x, y) in src]
    coeffs = find_perspective_coeffs(src, dst)
    result = img.transform(img.size, Image.Transform.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
    print('Perspective applied')
    return result

def find_perspective_coeffs(src, dst):
    matrix = []
    for p1, p2 in zip(dst, src):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(src).reshape(8)
    res = np.dot(np.linalg.pinv(A), B)
    return np.array(res).reshape(8)

def random_crop(img, crop_ratio=0.9):
    print('Applying random crop')
    width, height = img.size
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    x = random.randint(0, width - crop_w)
    y = random.randint(0, height - crop_h)
    result = img.crop((x, y, x + crop_w, y + crop_h)).resize((width, height), resample=Image.Resampling.LANCZOS)
    print('Crop applied')
    return result

def random_brightness_contrast(img):
    print('Applying random brightness/contrast')
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    print('Brightness/contrast applied')
    return img

def draw_board_with_pieces(fen, piece_images, square_size, out_path, colors=None):
    print('Drawing board with pieces')
    board_img = Image.new('RGBA', (BOARD_SIZE * square_size, BOARD_SIZE * square_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(board_img)
    if colors is None:
        colors = [(240, 217, 181), (181, 136, 99)]  # Default colors
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
                    piece_resized = piece_img.resize((square_size, square_size), resample=Image.Resampling.LANCZOS).convert('RGBA')
                    try:
                        board_img.paste(piece_resized, (x, y), piece_resized)
                    except Exception as e:
                        print(f"Paste error at rank {rank_idx}, file {file_idx}, char '{char}': {e}")
                        print(f"piece_img.size: {piece_img.size}, piece_resized.size: {piece_resized.size}, board_img.size: {board_img.size}")
                        raise
                file_idx += 1
    print('Board drawing complete')
    return board_img

def main():
    print('Main started')
    if not os.path.exists(OUTPUT_DIR):
        print(f'Creating output dir: {OUTPUT_DIR}')
        os.makedirs(OUTPUT_DIR)
    for i in range(10):
        try:
            print(f'Generating board {i}')
            set_path = random.choice(CHESS_SET_PATHS)
            print(f'Using chess set: {set_path}')
            piece_images = extract_piece_images(set_path)
            square_size = random.randint(*SQUARE_SIZE_RANGE)
            print(f'Using square size: {square_size}')
            board_img = draw_board_with_pieces(START_FEN, piece_images, square_size, None)
            if random.random() < 0.7:
                board_img = random_crop(board_img, crop_ratio=random.uniform(0.85, 0.98))
            if random.random() < 0.7:
                board_img = random_perspective(board_img)
            if random.random() < 0.7:
                board_img = random_brightness_contrast(board_img)
            out_path = OUTPUT_IMAGE_TEMPLATE.format(i)
            print(f'Saving to {out_path}')
            board_img.save(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Error generating board {i}: {e}")
    print('Main finished')

if __name__ == '__main__':
    main() 