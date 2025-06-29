#!/usr/bin/env python3
"""
Enhanced Chess Board Screenshot Generator
Generates random chess positions, displays them in GUI, takes screenshots,
and saves them with PGN files and a summary report.
"""

import os
import uuid
import time
from datetime import datetime
from PIL import ImageGrab
from chess_board import ChessBoard
from pgn_generator import PGNGenerator

def generate_board_screenshots(num_positions: int = 10, piece_style: str = "standard", 
                             output_dir: str = "boards") -> None:
    """
    Generate chess board screenshots with different piece styles.
    
    Args:
        num_positions: Number of positions to generate
        piece_style: Chess piece style to use ("standard", "outline", "filled", "simple", "circled", "bold")
        output_dir: Directory to save screenshots and files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generators
    pgn_gen = PGNGenerator()
    board = ChessBoard(piece_style=piece_style)
    
    # Generate report content
    report_lines = [
        f"Chess Board Screenshots Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Piece Style: {piece_style}",
        f"Number of Positions: {num_positions}",
        f"Output Directory: {output_dir}",
        f"{'='*60}",
        f""
    ]
    
    print(f"Generating {num_positions} chess positions with {piece_style} style...")
    print(f"Output directory: {output_dir}")
    print()
    
    for i in range(num_positions):
        # Generate random position
        position = pgn_gen.generate_random_position()
        
        # Create unique ID for this position
        position_id = str(uuid.uuid4())
        
        # Set up the board
        board.set_position(position)
        
        # Display in GUI
        title = f"Chess Position {i+1} - {piece_style.title()} Style"
        board.display_gui(title)
        
        # Wait for GUI to render
        time.sleep(0.5)
        
        # Take screenshot
        screenshot_path = os.path.join(output_dir, f"chess_board_{position_id}.png")
        pgn_path = os.path.join(output_dir, f"chess_board_{position_id}.pgn")
        
        # Capture screenshot with improved positioning
        try:
            # Get window position and size
            if board.root:
                # Wait a bit more for rendering
                time.sleep(0.2)
                
                # Capture with adjusted coordinates
                # Offset: 25 pixels right, 20 pixels down
                # Size: increase by 25% width, 25% height, then reduce width by 30 pixels
                x_offset = 25
                y_offset = 20
                width_increase = 0.25
                height_increase = 0.25
                width_reduction = 30
                
                # Calculate capture area
                window_width = board.board_size + 100
                window_height = board.board_size + 100
                
                capture_width = int(window_width * (1 + width_increase)) - width_reduction
                capture_height = int(window_height * (1 + height_increase))
                
                # Capture the area
                screenshot = ImageGrab.grab(bbox=(
                    x_offset, y_offset, 
                    x_offset + capture_width, 
                    y_offset + capture_height
                ))
                
                screenshot.save(screenshot_path)
                print(f"Position {i+1}: Screenshot saved as {screenshot_path}")
                print(f"  Capture area: {capture_width}x{capture_height} pixels")
                print(f"  Capture coordinates: ({x_offset}, {y_offset}) to ({x_offset + capture_width}, {y_offset + capture_height})")
                
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            screenshot_path = "ERROR"
        
        # Save PGN
        try:
            with open(pgn_path, 'w') as f:
                f.write(f"[Event \"Generated Position\"]\n")
                f.write(f"[Date \"{datetime.now().strftime('%Y.%m.%d')}\"]\n")
                f.write(f"[Position \"{position}\"]\n")
                f.write(f"[PieceStyle \"{piece_style}\"]\n")
                f.write(f"\n{position}\n")
            print(f"  PGN saved as {pgn_path}")
        except Exception as e:
            print(f"Error saving PGN: {e}")
            pgn_path = "ERROR"
        
        # Add to report
        report_lines.extend([
            f"Position {i+1}:",
            f"  ID: {position_id}",
            f"  PGN: {position}",
            f"  Screenshot: chess_board_{position_id}.png",
            f"  PGN File: chess_board_{position_id}.pgn",
            f""
        ])
        
        # Close GUI
        board.close_gui()
        
        # Small delay between positions
        time.sleep(0.2)
    
    # Save summary report
    report_path = os.path.join(output_dir, f"chess_screenshots_report_{piece_style}.txt")
    try:
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nSummary report saved as {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    print(f"\nGenerated {num_positions} chess positions with {piece_style} style!")
    print(f"Check the '{output_dir}' folder for screenshots and PGN files.")

def main():
    """Main function to run the screenshot generator."""
    print("Enhanced Chess Board Screenshot Generator")
    print("=" * 50)
    
    # Available piece styles
    styles = ["standard", "outline", "filled", "simple", "circled", "bold"]
    
    print("Available piece styles:")
    for i, style in enumerate(styles, 1):
        print(f"  {i}. {style}")
    
    print()
    
    # Generate with different styles
    for style in styles:
        print(f"\nGenerating positions with '{style}' style...")
        generate_board_screenshots(num_positions=3, piece_style=style)
        print(f"Completed '{style}' style generation.")
    
    print("\nAll styles completed! Check the 'boards' folder for results.")

if __name__ == "__main__":
    main() 