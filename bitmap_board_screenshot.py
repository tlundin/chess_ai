import os
import uuid
import time
from datetime import datetime
from pgn_generator import PGNGenerator
from PIL import Image, ImageDraw, PngImagePlugin

# Board and piece settings
BOARD_SIZE = 8
PIECE_IMAGE = 'chess_pieces.png'
OUTPUT_DIR = 'boards'

# Piece order in the image (left to right)
PIECE_ORDER = ['K', 'Q', 'B', 'N', 'R', 'P']

# Map FEN letters to piece image index (row, col)
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
    # Add PGN as metadata if provided
    if pgn_text is not None:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pgn", pgn_text)
        board_img.save(out_path, pnginfo=meta)
    else:
        board_img.save(out_path)

def generate_bitmap_board_screenshots(num_positions=10, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    pgn_gen = PGNGenerator()
    piece_images, piece_size = extract_piece_images(PIECE_IMAGE)
    report_lines = [
        f"Bitmap Chess Board Screenshots Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of Positions: {num_positions}",
        f"Output Directory: {output_dir}",
        f"{'='*60}",
        f""
    ]
    print(f"Generating {num_positions} bitmap chess positions...")
    print(f"Output directory: {output_dir}")
    print()
    for i in range(num_positions):
        position = pgn_gen.generate_random_position()
        position_id = str(uuid.uuid4())
        png_path = os.path.join(output_dir, f"chess_board_{position_id}.png")
        pgn_path = os.path.join(output_dir, f"chess_board_{position_id}.pgn")
        draw_board_with_pieces(position, piece_images, piece_size, png_path, pgn_text=position)
        print(f"Position {i+1}: Image saved as {png_path}")
        try:
            with open(pgn_path, 'w') as f:
                f.write(f"[Event \"Generated Position\"]\n")
                f.write(f"[Date \"{datetime.now().strftime('%Y.%m.%d')}\"]\n")
                f.write(f"[Position \"{position}\"]\n")
                f.write(f"\n{position}\n")
            print(f"  PGN saved as {pgn_path}")
        except Exception as e:
            print(f"Error saving PGN: {e}")
            pgn_path = "ERROR"
        report_lines.extend([
            f"Position {i+1}:",
            f"  ID: {position_id}",
            f"  PGN: {position}",
            f"  Image: chess_board_{position_id}.png",
            f"  PGN File: chess_board_{position_id}.pgn",
            f""
        ])
        time.sleep(0.1)
    report_path = os.path.join(output_dir, f"bitmap_chess_screenshots_report.txt")
    try:
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nSummary report saved as {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
    print(f"\nGenerated {num_positions} bitmap chess positions!")
    print(f"Check the '{output_dir}' folder for images and PGN files.")

def main():
    print("Bitmap Chess Board Screenshot Generator")
    print("=" * 50)
    generate_bitmap_board_screenshots(num_positions=10)

if __name__ == '__main__':
    main() 