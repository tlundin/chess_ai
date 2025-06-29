#!/usr/bin/env python3
"""
Chess Piece Styles Demo
Demonstrates different chess piece styles available in the chess board.
"""

from chess_board import ChessBoard
import time

def demo_chess_styles():
    """Demonstrate all available chess piece styles."""
    print("Chess Piece Styles Demo")
    print("=" * 40)
    
    # Create a chess board with a test position
    board = ChessBoard()
    
    # Set up a position with all piece types
    test_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    board.set_position(test_position)
    
    # Get available styles
    styles = board.get_available_styles()
    print(f"Available styles: {', '.join(styles)}")
    print()
    
    # Demo each style
    for style in styles:
        print(f"Style: {style.upper()}")
        print("-" * 20)
        
        # Set the style
        board.set_piece_style(style)
        
        # Show in console
        board.display()
        
        # Show in GUI
        board.display_gui(f"Chess Style: {style.title()}")
        
        # Wait a moment to see the GUI
        time.sleep(2)
        
        # Close GUI
        board.close_gui()
        
        print()
        print("=" * 40)
        print()

def demo_custom_position_with_style():
    """Demo a custom position with a specific style."""
    print("Custom Position with Bold Style")
    print("=" * 40)
    
    # Create board with bold style
    board = ChessBoard(piece_style="bold")
    
    # Set a custom position
    custom_position = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
    board.set_position(custom_position)
    
    # Display
    board.display()
    board.display_gui("Custom Position - Bold Style")
    
    print("GUI will stay open. Close it to continue...")
    if board.root:
        board.root.mainloop()

def main():
    """Run the chess styles demo."""
    print("Chess Piece Styles Demonstration")
    print("=" * 50)
    print()
    
    # Demo 1: Show all styles
    demo_chess_styles()
    
    # Demo 2: Custom position with specific style
    demo_custom_position_with_style()

if __name__ == "__main__":
    main() 