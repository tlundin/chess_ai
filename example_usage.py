#!/usr/bin/env python3
"""
Example usage of the ChessBoard class with graphical interface.
This script demonstrates how to use the chess board with both console and GUI display.
"""

from chess_board import ChessBoard
import time

def main():
    """Demonstrate the chess board functionality."""
    print("Chess Board with Graphical Interface")
    print("=" * 40)
    
    # Create a chess board instance
    board = ChessBoard()
    
    # Example 1: Starting position
    print("\n1. Displaying starting position...")
    board.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    
    # Show in console
    board.display()
    
    # Show in GUI
    board.display_gui("Chess - Starting Position")
    
    # Wait a moment to see the GUI
    time.sleep(2)
    
    # Example 2: After 1.e4
    print("\n2. After 1.e4...")
    board.set_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    board.display()
    board.display_gui("Chess - After 1.e4")
    
    time.sleep(2)
    
    # Example 3: After 1.e4 e5
    print("\n3. After 1.e4 e5...")
    board.set_position("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR")
    board.display()
    board.display_gui("Chess - After 1.e4 e5")
    
    time.sleep(2)
    
    # Example 4: Custom position - Scholar's Mate
    print("\n4. Scholar's Mate position...")
    board.set_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR")
    board.display()
    board.display_gui("Chess - Scholar's Mate")
    
    time.sleep(2)
    
    # Example 5: Interactive piece placement
    print("\n5. Setting up a custom position piece by piece...")
    board.clear_board()
    
    # Place kings
    board.set_piece_at('e1', 'K')  # White king
    board.set_piece_at('e8', 'k')  # Black king
    
    # Place queens
    board.set_piece_at('d1', 'Q')  # White queen
    board.set_piece_at('d8', 'q')  # Black queen
    
    # Place rooks
    board.set_piece_at('a1', 'R')  # White rook
    board.set_piece_at('h8', 'r')  # Black rook
    
    # Place bishops
    board.set_piece_at('c1', 'B')  # White bishop
    board.set_piece_at('f8', 'b')  # Black bishop
    
    # Place knights
    board.set_piece_at('b1', 'N')  # White knight
    board.set_piece_at('g8', 'n')  # Black knight
    
    # Place some pawns
    board.set_piece_at('e2', 'P')  # White pawn
    board.set_piece_at('e7', 'p')  # Black pawn
    
    board.display()
    board.display_gui("Chess - Custom Setup")
    
    # Example 6: Get information about the position
    print(f"\n6. Position information:")
    print(f"   FEN: {board.get_fen()}")
    print(f"   Piece at e1: {board.get_piece_at('e1')}")
    print(f"   Piece at e8: {board.get_piece_at('e8')}")
    print(f"   Piece at a1: {board.get_piece_at('a1')}")
    print(f"   Piece at h8: {board.get_piece_at('h8')}")
    
    # Keep the GUI open for the last position
    print("\nGUI window will stay open. Close it to exit.")
    if board.root:
        board.root.mainloop()

if __name__ == "__main__":
    main() 