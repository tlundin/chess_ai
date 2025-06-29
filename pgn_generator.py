#!/usr/bin/env python3
"""
PGN Generator for Chess Positions
Generates valid chess positions in PGN/FEN format.
"""

import random
from typing import List, Dict, Tuple

class PGNGenerator:
    """Generates valid chess positions in PGN format."""
    
    def __init__(self):
        # Standard piece counts for a valid chess position
        self.max_pieces = {
            'K': 1, 'Q': 9, 'R': 10, 'B': 10, 'N': 10, 'P': 8,  # White pieces
            'k': 1, 'q': 9, 'r': 10, 'b': 10, 'n': 10, 'p': 8   # Black pieces
        }
        
        # Starting positions for pieces (for realistic positions)
        self.starting_positions = {
            'K': ['e1'], 'Q': ['d1'], 'R': ['a1', 'h1'], 'B': ['c1', 'f1'], 'N': ['b1', 'g1'], 'P': ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'],
            'k': ['e8'], 'q': ['d8'], 'r': ['a8', 'h8'], 'b': ['c8', 'f8'], 'n': ['b8', 'g8'], 'p': ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']
        }
    
    def generate_random_position(self) -> str:
        """Generate a random valid chess position."""
        # Start with empty board
        board = [[' ' for _ in range(8)] for _ in range(8)]
        
        # Always place kings first (required for valid position)
        self._place_kings(board)
        
        # Place other pieces randomly
        self._place_other_pieces(board)
        
        # Convert to PGN format
        return self._board_to_pgn(board)
    
    def generate_starting_position(self) -> str:
        """Generate the standard starting position."""
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    
    def generate_common_positions(self) -> List[str]:
        """Generate a list of common chess positions."""
        positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            
            # After 1.e4
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
            
            # After 1.e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR",
            
            # After 1.e4 e5 2.Nf3
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
            
            # After 1.e4 e5 2.Nf3 Nc6
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
            
            # After 1.e4 e5 2.Nf3 Nc6 3.Bb5
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
            
            # Scholar's Mate position
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR",
            
            # Fool's Mate position
            "rnb1kbnr/pppp1ppp/8/6q1/5P2/8/PPPPP1PP/RNBQKBNR",
            
            # Endgame: King and Queen vs King
            "8/8/8/8/8/8/4K3/4Q3",
            
            # Endgame: King and Rook vs King
            "8/8/8/8/8/8/4K3/4R3",
            
            # Complex middlegame position
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
            
            # Position with many pieces
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
            
            # Position with queens
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            
            # Position with bishops
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            
            # Position with knights
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        ]
        return positions
    
    def generate_random_game_position(self, max_moves: int = 20) -> str:
        """Generate a position that might occur in a real game."""
        # Start with starting position
        board = self._pgn_to_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        
        # Make some random moves
        for _ in range(random.randint(5, max_moves)):
            self._make_random_move(board)
        
        return self._board_to_pgn(board)
    
    def _place_kings(self, board: List[List[str]]) -> None:
        """Place kings on the board (required for valid position)."""
        # Place white king
        white_king_pos = random.choice(['e1', 'd1', 'f1', 'e2', 'd2', 'f2'])
        file_idx = ord(white_king_pos[0]) - ord('a')
        rank_idx = 8 - int(white_king_pos[1])
        board[rank_idx][file_idx] = 'K'
        
        # Place black king (not adjacent to white king)
        black_king_pos = random.choice(['e8', 'd8', 'f8', 'e7', 'd7', 'f7'])
        file_idx = ord(black_king_pos[0]) - ord('a')
        rank_idx = 8 - int(black_king_pos[1])
        board[rank_idx][file_idx] = 'k'
    
    def _place_other_pieces(self, board: List[List[str]]) -> None:
        """Place other pieces randomly on the board."""
        pieces = ['Q', 'R', 'B', 'N', 'P', 'q', 'r', 'b', 'n', 'p']
        
        for piece in pieces:
            # Determine how many of this piece to place
            max_count = self.max_pieces[piece]
            count = random.randint(0, min(max_count, 4))  # Limit to 4 of each piece
            
            for _ in range(count):
                # Find empty square
                empty_squares = []
                for rank in range(8):
                    for file in range(8):
                        if board[rank][file] == ' ':
                            empty_squares.append((rank, file))
                
                if empty_squares:
                    rank, file = random.choice(empty_squares)
                    board[rank][file] = piece
    
    def _make_random_move(self, board: List[List[str]]) -> None:
        """Make a random move on the board."""
        # Find all pieces
        pieces = []
        for rank in range(8):
            for file in range(8):
                if board[rank][file] != ' ':
                    pieces.append((rank, file, board[rank][file]))
        
        if not pieces:
            return
        
        # Pick a random piece
        rank, file, piece = random.choice(pieces)
        
        # Find possible moves (simplified - just random empty squares)
        empty_squares = []
        for r in range(8):
            for f in range(8):
                if board[r][f] == ' ':
                    empty_squares.append((r, f))
        
        if empty_squares:
            new_rank, new_file = random.choice(empty_squares)
            board[new_rank][new_file] = piece
            board[rank][file] = ' '
    
    def _board_to_pgn(self, board: List[List[str]]) -> str:
        """Convert board array to PGN format."""
        pgn_ranks = []
        
        for rank in board:
            pgn_rank = ""
            empty_count = 0
            
            for piece in rank:
                if piece == ' ':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        pgn_rank += str(empty_count)
                        empty_count = 0
                    pgn_rank += piece
            
            if empty_count > 0:
                pgn_rank += str(empty_count)
            
            pgn_ranks.append(pgn_rank)
        
        return '/'.join(pgn_ranks)
    
    def _pgn_to_board(self, pgn: str) -> List[List[str]]:
        """Convert PGN format to board array."""
        board = [[' ' for _ in range(8)] for _ in range(8)]
        ranks = pgn.split('/')
        
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    if file_idx < 8:
                        board[rank_idx][file_idx] = char
                        file_idx += 1
        
        return board
    
    def generate_position_set(self, count: int = 100) -> List[str]:
        """Generate a set of diverse chess positions."""
        positions = []
        
        # Add some common positions
        common_positions = self.generate_common_positions()
        positions.extend(common_positions[:min(count//4, len(common_positions))])
        
        # Add random positions
        remaining = count - len(positions)
        for _ in range(remaining):
            if random.random() < 0.7:  # 70% random positions
                positions.append(self.generate_random_position())
            else:  # 30% game-like positions
                positions.append(self.generate_random_game_position())
        
        return positions[:count]


def main():
    """Test the PGN generator."""
    generator = PGNGenerator()
    
    print("PGN Generator Test")
    print("=" * 30)
    
    # Test 1: Starting position
    print("\n1. Starting position:")
    start_pos = generator.generate_starting_position()
    print(start_pos)
    
    # Test 2: Random position
    print("\n2. Random position:")
    random_pos = generator.generate_random_position()
    print(random_pos)
    
    # Test 3: Game-like position
    print("\n3. Game-like position:")
    game_pos = generator.generate_random_game_position()
    print(game_pos)
    
    # Test 4: Generate position set
    print("\n4. Generating 10 positions:")
    positions = generator.generate_position_set(10)
    for i, pos in enumerate(positions, 1):
        print(f"{i:2d}. {pos}")


if __name__ == "__main__":
    main() 