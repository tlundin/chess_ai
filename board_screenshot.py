#!/usr/bin/env python3
"""
Chess Board Screenshot Generator
Generates random chess positions, displays them in GUI, and saves screenshots.
"""

import os
import uuid
import time
from typing import List, Tuple
from pgn_generator import PGNGenerator
from chess_board import ChessBoard
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import io

class BoardScreenshotGenerator:
    """Generates screenshots of chess board positions."""
    
    def __init__(self):
        self.pgn_generator = PGNGenerator()
        self.chess_board = ChessBoard()
        self.output_folder = "boards"
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
    
    def generate_and_screenshot(self, count: int = 10) -> List[Tuple[str, str, str]]:
        """Generate random positions and take screenshots."""
        print(f"Generating {count} chess board screenshots...")
        print("=" * 50)
        
        results = []
        
        for i in range(count):
            print(f"\nProcessing position {i+1}/{count}...")
            
            # Generate random position
            position = self.pgn_generator.generate_random_position()
            print(f"Generated position: {position}")
            
            # Set position and display GUI
            self.chess_board.set_position(position)
            self.chess_board.display_gui(f"Chess Position {i+1}")
            
            # Wait for GUI to render
            time.sleep(0.5)
            
            # Take screenshot
            screenshot_path = self._take_screenshot(position, i+1)
            
            if screenshot_path:
                results.append((position, screenshot_path, str(uuid.uuid4())))
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Failed to take screenshot")
            
            # Close GUI window
            self.chess_board.close_gui()
            time.sleep(0.2)  # Brief pause between positions
        
        return results
    
    def _take_screenshot(self, position: str, position_number: int) -> str | None:
        """Take a screenshot of the current chess board window."""
        if not self.chess_board.root:
            return None
        
        try:
            # Wait a bit more for the window to fully render
            self.chess_board.root.update()
            time.sleep(0.3)
            
            # Method 1: Try to capture the exact window
            try:
                # Get window coordinates with better precision
                x = self.chess_board.root.winfo_rootx()
                y = self.chess_board.root.winfo_rooty()
                width = self.chess_board.root.winfo_width()
                height = self.chess_board.root.winfo_height()
                
                # Add some padding to ensure we capture the full window
                padding = 10
                x -= padding
                y -= padding
                width += 2 * padding
                height += 2 * padding
                
                print(f"Method 1: Capturing window at ({x}, {y}) with size {width}x{height}")
                
                # Take screenshot of the window area
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                
            except Exception as e:
                print(f"Method 1 failed: {e}")
                # Method 2: Capture a larger area and crop
                print("Trying Method 2: Larger area capture...")
                
                # Get window center
                center_x = self.chess_board.root.winfo_rootx() + self.chess_board.root.winfo_width() // 2
                center_y = self.chess_board.root.winfo_rooty() + self.chess_board.root.winfo_height() // 2
                
                # Capture a larger area around the window
                capture_size = 800  # Larger capture area
                x = center_x - capture_size // 2
                y = center_y - capture_size // 2
                width = capture_size
                height = capture_size
                
                print(f"Method 2: Capturing area at ({x}, {y}) with size {width}x{height}")
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            
            # Generate filename with UUID
            filename = f"chess_board_{uuid.uuid4()}.png"
            filepath = os.path.join(self.output_folder, filename)
            
            # Save screenshot
            screenshot.save(filepath, "PNG")
            
            # Also save a text file with the PGN
            pgn_filename = filename.replace('.png', '.pgn')
            pgn_filepath = os.path.join(self.output_folder, pgn_filename)
            
            with open(pgn_filepath, 'w') as f:
                f.write(f"Position {position_number}\n")
                f.write(f"PGN: {position}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Window size: {width}x{height}\n")
            
            return filepath
            
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None
    
    def generate_with_custom_positions(self, positions: List[str]) -> List[Tuple[str, str, str]]:
        """Generate screenshots for specific positions."""
        print(f"Generating screenshots for {len(positions)} custom positions...")
        print("=" * 50)
        
        results = []
        
        for i, position in enumerate(positions):
            print(f"\nProcessing position {i+1}/{len(positions)}...")
            print(f"Position: {position}")
            
            # Set position and display GUI
            self.chess_board.set_position(position)
            self.chess_board.display_gui(f"Custom Position {i+1}")
            
            # Wait for GUI to render
            time.sleep(0.5)
            
            # Take screenshot
            screenshot_path = self._take_screenshot(position, i+1)
            
            if screenshot_path:
                results.append((position, screenshot_path, str(uuid.uuid4())))
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print("Failed to take screenshot")
            
            # Close GUI window
            self.chess_board.close_gui()
            time.sleep(0.2)
        
        return results
    
    def create_summary_report(self, results: List[Tuple[str, str, str]]) -> None:
        """Create a summary report of all generated screenshots."""
        report_path = os.path.join(self.output_folder, "screenshot_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Chess Board Screenshot Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total screenshots: {len(results)}\n\n")
            
            for i, (position, screenshot_path, uuid_str) in enumerate(results, 1):
                f.write(f"Position {i}:\n")
                f.write(f"  PGN: {position}\n")
                f.write(f"  Screenshot: {os.path.basename(screenshot_path)}\n")
                f.write(f"  UUID: {uuid_str}\n")
                f.write(f"  Full path: {screenshot_path}\n")
                f.write("\n")
        
        print(f"\nSummary report saved: {report_path}")


def main():
    """Main function to generate chess board screenshots."""
    generator = BoardScreenshotGenerator()
    
    # Generate 10 random positions
    print("Chess Board Screenshot Generator")
    print("=" * 40)
    
    results = generator.generate_and_screenshot(10)
    
    # Create summary report
    generator.create_summary_report(results)
    
    print(f"\nCompleted! Generated {len(results)} screenshots.")
    print(f"All files saved in the '{generator.output_folder}' folder.")
    
    # Show some sample results
    print(f"\nSample results:")
    for i, (position, screenshot_path, uuid_str) in enumerate(results[:3], 1):
        print(f"{i}. {os.path.basename(screenshot_path)} - {position}")


if __name__ == "__main__":
    main() 