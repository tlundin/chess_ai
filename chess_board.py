import re
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

class ChessBoard:
    """A chess board that can display positions from PGN notation with graphical interface."""
    
    def __init__(self, piece_style: str = "standard"):
        self.board = self._create_empty_board()
        
        # Different chess piece styles
        self.piece_styles = {
            "standard": {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White pieces
                'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'   # Black pieces
            },
            "outline": {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White pieces (outlined)
                'k': '♔', 'q': '♕', 'r': '♖', 'b': '♗', 'n': '♘', 'p': '♙'   # Black pieces (same symbols, different color)
            },
            "filled": {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White pieces
                'k': '♔', 'q': '♕', 'r': '♖', 'b': '♗', 'n': '♘', 'p': '♙'   # Black pieces (same symbols, different color)
            },
            "simple": {
                'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',  # White pieces (letters)
                'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p'   # Black pieces (letters)
            },
            "circled": {
                'K': 'Ⓚ', 'Q': 'Ⓠ', 'R': 'Ⓡ', 'B': 'Ⓑ', 'N': 'Ⓝ', 'P': 'Ⓟ',  # White pieces (circled letters)
                'k': 'ⓚ', 'q': 'ⓠ', 'r': 'ⓡ', 'b': 'ⓑ', 'n': 'ⓝ', 'p': 'ⓟ'   # Black pieces (circled letters)
            },
            "bold": {
                'K': '𝐊', 'Q': '𝐐', 'R': '𝐑', 'B': '𝐁', 'N': '𝐍', 'P': '𝐏',  # White pieces (bold letters)
                'k': '𝐤', 'q': '𝐪', 'r': '𝐫', 'b': '𝐛', 'n': '𝐧', 'p': '𝐩'   # Black pieces (bold letters)
            }
        }
        
        # Set the piece style
        self.piece_style = piece_style
        self.piece_symbols = self.piece_styles.get(piece_style, self.piece_styles["standard"])
        
        self.root = None
        self.canvas = None
        self.square_size = 60
        self.board_size = self.square_size * 8
        
    def _create_empty_board(self) -> List[List[str]]:
        """Create an empty 8x8 chess board."""
        return [[' ' for _ in range(8)] for _ in range(8)]
    
    def _parse_fen_position(self, fen_position: str) -> None:
        """Parse a FEN position string and set up the board."""
        # Split FEN into parts (we only need the first part for piece placement)
        parts = fen_position.split()
        piece_placement = parts[0]
        
        # Clear the board
        self.board = self._create_empty_board()
        
        # Parse piece placement
        ranks = piece_placement.split('/')
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    if file_idx < 8:
                        self.board[rank_idx][file_idx] = char
                        file_idx += 1
    
    def _parse_pgn_position(self, pgn_position: str) -> None:
        """Parse a PGN position string and convert to FEN format."""
        # Remove any extra whitespace and newlines
        pgn_position = re.sub(r'\s+', ' ', pgn_position.strip())
        
        # Split into ranks
        ranks = pgn_position.split('/')
        if len(ranks) != 8:
            raise ValueError("Invalid PGN position: must have exactly 8 ranks")
        
        # Convert PGN to FEN format
        fen_ranks = []
        for rank in ranks:
            fen_rank = ""
            empty_count = 0
            
            for char in rank:
                if char.isdigit():
                    empty_count += int(char)
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += char
            
            if empty_count > 0:
                fen_rank += str(empty_count)
            
            fen_ranks.append(fen_rank)
        
        fen_position = '/'.join(fen_ranks)
        self._parse_fen_position(fen_position)
    
    def set_position(self, position: str) -> None:
        """Set the board position from PGN or FEN notation."""
        if '/' in position and len(position.split('/')) == 8:
            # Looks like PGN or FEN format
            if any(char.isdigit() for char in position if char not in '12345678/'):
                # Contains numbers > 8, likely FEN
                self._parse_fen_position(position)
            else:
                # Likely PGN format
                self._parse_pgn_position(position)
        else:
            raise ValueError("Invalid position format. Use PGN or FEN notation.")
    
    def display(self) -> None:
        """Display the current board position in console."""
        print("  a  b  c  d  e  f  g  h")
        print("  -----------------------")
        
        for rank in range(8):
            print(f"{8-rank} ", end="")
            for file in range(8):
                piece = self.board[rank][file]
                if piece == ' ':
                    # Empty square - alternate colors
                    if (rank + file) % 2 == 0:
                        print(" . ", end="")
                    else:
                        print(" # ", end="")
                else:
                    symbol = self.piece_symbols.get(piece, piece)
                    print(f"{symbol} ", end="")
            print(f" {8-rank}")
        
        print("  -----------------------")
        print("  a  b  c  d  e  f  g  h")
    
    def display_gui(self, title: str = "Chess Board") -> None:
        """Display the chess board in a graphical window."""
        if self.root is None:
            self.root = tk.Tk()
            self.root.title(title)
            self.root.resizable(False, False)
            
            # Create canvas
            self.canvas = tk.Canvas(
                self.root, 
                width=self.board_size + 100, 
                height=self.board_size + 100,
                bg='white'
            )
            self.canvas.pack(padx=10, pady=10)
            
            # Bind close event
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self._draw_board()
        self.root.update()
    
    def _draw_board(self) -> None:
        """Draw the chess board and pieces on the canvas."""
        if self.canvas is None:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw board squares
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size + 50
                y1 = rank * self.square_size + 50
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Alternate colors for squares
                color = "#F0D9B5" if (rank + file) % 2 == 0 else "#B58863"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        # Draw file labels (a-h)
        for file in range(8):
            x = file * self.square_size + 50 + self.square_size // 2
            y = self.board_size + 70
            label = chr(ord('a') + file)
            self.canvas.create_text(x, y, text=label, font=("Arial", 12, "bold"))
        
        # Draw rank labels (1-8)
        for rank in range(8):
            x = 30
            y = rank * self.square_size + 50 + self.square_size // 2
            label = str(8 - rank)
            self.canvas.create_text(x, y, text=label, font=("Arial", 12, "bold"))
        
        # Draw pieces
        for rank in range(8):
            for file in range(8):
                piece = self.board[rank][file]
                if piece != ' ':
                    x = file * self.square_size + 50 + self.square_size // 2
                    y = rank * self.square_size + 50 + self.square_size // 2
                    symbol = self.piece_symbols.get(piece, piece)
                    
                    # Color based on piece color
                    color = "white" if piece.isupper() else "black"
                    self.canvas.create_text(
                        x, y, 
                        text=symbol, 
                        font=("Arial", 36, "bold"),
                        fill=color
                    )
    
    def _on_closing(self) -> None:
        """Handle window closing."""
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
    
    def close_gui(self) -> None:
        """Close the graphical window."""
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
    
    def get_piece_at(self, square: str) -> Optional[str]:
        """Get the piece at a given square (e.g., 'e4')."""
        if len(square) != 2:
            return None
        
        file_char, rank_char = square[0].lower(), square[1]
        
        if not (file_char in 'abcdefgh' and rank_char in '12345678'):
            return None
        
        file_idx = ord(file_char) - ord('a')
        rank_idx = 8 - int(rank_char)
        
        if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
            return self.board[rank_idx][file_idx]
        return None
    
    def set_piece_at(self, square: str, piece: str) -> bool:
        """Set a piece at a given square (e.g., 'e4', 'K')."""
        if len(square) != 2:
            return False
        
        file_char, rank_char = square[0].lower(), square[1]
        
        if not (file_char in 'abcdefgh' and rank_char in '12345678'):
            return False
        
        file_idx = ord(file_char) - ord('a')
        rank_idx = 8 - int(rank_char)
        
        if 0 <= rank_idx < 8 and 0 <= file_idx < 8:
            self.board[rank_idx][file_idx] = piece
            # Update GUI if it's open
            if self.canvas:
                self._draw_board()
            return True
        return False
    
    def clear_board(self) -> None:
        """Clear the board (set all squares to empty)."""
        self.board = self._create_empty_board()
        if self.canvas:
            self._draw_board()
    
    def get_fen(self) -> str:
        """Get the current position in FEN format."""
        fen_ranks = []
        
        for rank in self.board:
            fen_rank = ""
            empty_count = 0
            
            for piece in rank:
                if piece == ' ':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += piece
            
            if empty_count > 0:
                fen_rank += str(empty_count)
            
            fen_ranks.append(fen_rank)
        
        return '/'.join(fen_ranks)
    
    def set_piece_style(self, style: str) -> None:
        """Change the chess piece style."""
        if style in self.piece_styles:
            self.piece_style = style
            self.piece_symbols = self.piece_styles[style]
            # Update GUI if it's open
            if self.canvas:
                self._draw_board()
        else:
            available_styles = ", ".join(self.piece_styles.keys())
            raise ValueError(f"Invalid style '{style}'. Available styles: {available_styles}")
    
    def get_available_styles(self) -> List[str]:
        """Get list of available piece styles."""
        return list(self.piece_styles.keys())


def main():
    """Example usage of the ChessBoard class."""
    board = ChessBoard()
    
    print("Chess Board Position Display")
    print("=" * 30)
    
    # Example 1: Starting position
    print("\n1. Starting position:")
    board.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    board.display()
    board.display_gui("Starting Position")
    
    # Example 2: After 1.e4
    print("\n2. After 1.e4:")
    board.set_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    board.display()
    board.display_gui("After 1.e4")
    
    # Example 3: Custom position
    print("\n3. Custom position (checkmate in 1):")
    board.set_position("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR")
    board.display()
    board.display_gui("Checkmate Position")
    
    # Example 4: Using individual piece placement
    print("\n4. Setting up a position piece by piece:")
    board.clear_board()
    board.set_piece_at('e1', 'K')  # White king
    board.set_piece_at('e8', 'k')  # Black king
    board.set_piece_at('d1', 'Q')  # White queen
    board.set_piece_at('d8', 'q')  # Black queen
    board.set_piece_at('a1', 'R')  # White rook
    board.set_piece_at('h8', 'r')  # Black rook
    board.display()
    board.display_gui("Custom Setup")
    
    # Example 5: Get FEN of current position
    print(f"\n5. FEN representation: {board.get_fen()}")
    
    # Example 6: Check piece at specific square
    print(f"\n6. Piece at e1: {board.get_piece_at('e1')}")
    print(f"   Piece at e8: {board.get_piece_at('e8')}")
    
    # Keep the GUI open
    if board.root:
        board.root.mainloop()


if __name__ == "__main__":
    main() 