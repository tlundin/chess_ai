# Chess Board Display

A Python chess board implementation that can display chess positions using both console output and a graphical interface. The board supports standard chess notation (PGN/FEN) and uses Unicode chess symbols for beautiful piece representation.

## Features

- **Unicode Chess Symbols**: Uses standard Unicode chess pieces (♔♕♖♗♘♙ for white, ♚♛♜♝♞♟ for black)
- **Graphical Interface**: Tkinter-based GUI with a proper chess board layout
- **Console Display**: Traditional ASCII-based display for terminal usage
- **Multiple Input Formats**: Supports both PGN and FEN notation
- **Interactive Piece Placement**: Add pieces individually to any square
- **Position Analysis**: Get FEN representation and query pieces at specific squares

## Installation

No external dependencies required! This project uses only Python's standard library:

- `tkinter` (GUI framework)
- `re` (regular expressions)
- `typing` (type hints)

## Usage

### Basic Usage

```python
from chess_board import ChessBoard

# Create a chess board
board = ChessBoard()

# Set a position using PGN notation
board.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

# Display in console
board.display()

# Display in GUI
board.display_gui("Starting Position")
```

### Console Output Example

```
  a  b  c  d  e  f  g  h
  -----------------------
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜  8
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟  7
6 .  #  .  #  .  #  .  #  6
5 #  .  #  .  #  .  #  .  5
4 .  #  .  #  .  #  .  #  4
3 #  .  #  .  #  .  #  .  3
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙  2
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖  1
  -----------------------
  a  b  c  d  e  f  g  h
```

### GUI Features

The graphical interface provides:
- **Proper Chess Board Colors**: Light and dark squares with authentic colors
- **File and Rank Labels**: a-h and 1-8 labels for easy square identification
- **Large Unicode Pieces**: Clear, readable chess symbols
- **Responsive Updates**: Board updates automatically when pieces are moved
- **Window Management**: Proper window closing and cleanup

### Interactive Piece Placement

```python
# Clear the board
board.clear_board()

# Place pieces individually
board.set_piece_at('e1', 'K')  # White king
board.set_piece_at('e8', 'k')  # Black king
board.set_piece_at('d1', 'Q')  # White queen

# The GUI updates automatically
```

### Position Analysis

```python
# Get FEN representation
fen = board.get_fen()
print(f"FEN: {fen}")

# Query pieces at specific squares
piece = board.get_piece_at('e1')  # Returns 'K' for white king
```

## Examples

Run the example script to see various chess positions:

```bash
python example_usage.py
```

This will demonstrate:
1. Starting position
2. After 1.e4
3. After 1.e4 e5
4. Scholar's Mate position
5. Custom piece placement
6. Position analysis

## File Structure

- `chess_board.py` - Main chess board implementation
- `example_usage.py` - Example usage and demonstrations
- `requirements.txt` - Dependencies (none required)
- `README.md` - This documentation

## Chess Notation

### PGN Format
Standard chess position notation where each rank is separated by `/`:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

### FEN Format
Extended chess notation that includes additional game state:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

### Piece Symbols
- **White pieces**: K (King), Q (Queen), R (Rook), B (Bishop), N (Knight), P (Pawn)
- **Black pieces**: k, q, r, b, n, p (lowercase)

## Contributing

Feel free to contribute improvements! Some ideas:
- Add move validation
- Implement game play functionality
- Add move history
- Support for different piece themes
- Export to image formats

## License

This project is open source and available under the MIT License. 