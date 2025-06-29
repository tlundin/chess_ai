#!/usr/bin/env python3
"""
Simple test script for the ChessBoard class.
"""

from chess_board import ChessBoard

def test_basic_functionality():
    """Test basic chess board functionality."""
    print("Testing Chess Board Functionality")
    print("=" * 40)
    
    board = ChessBoard()
    
    # Test 1: Starting position
    print("\n1. Testing starting position...")
    board.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    
    # Verify some key pieces
    assert board.get_piece_at('e1') == 'K', "White king should be at e1"
    assert board.get_piece_at('e8') == 'k', "Black king should be at e8"
    assert board.get_piece_at('a1') == 'R', "White rook should be at a1"
    assert board.get_piece_at('h8') == 'r', "Black rook should be at h8"
    print("✓ Starting position test passed")
    
    # Test 2: After 1.e4
    print("\n2. Testing after 1.e4...")
    board.set_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")
    
    assert board.get_piece_at('e4') == 'P', "White pawn should be at e4"
    assert board.get_piece_at('e2') == ' ', "e2 should be empty"
    print("✓ After 1.e4 test passed")
    
    # Test 3: Individual piece placement
    print("\n3. Testing individual piece placement...")
    board.clear_board()
    board.set_piece_at('e1', 'K')
    board.set_piece_at('e8', 'k')
    board.set_piece_at('d1', 'Q')
    
    assert board.get_piece_at('e1') == 'K', "White king should be at e1"
    assert board.get_piece_at('e8') == 'k', "Black king should be at e8"
    assert board.get_piece_at('d1') == 'Q', "White queen should be at d1"
    print("✓ Individual piece placement test passed")
    
    # Test 4: FEN generation
    print("\n4. Testing FEN generation...")
    fen = board.get_fen()
    expected_fen = "4k3/8/8/8/8/8/8/3QK3"
    assert fen == expected_fen, f"FEN should be {expected_fen}, got {fen}"
    print("✓ FEN generation test passed")
    
    # Test 5: Unicode symbols
    print("\n5. Testing Unicode symbols...")
    assert board.piece_symbols['K'] == '♔', "White king symbol should be ♔"
    assert board.piece_symbols['k'] == '♚', "Black king symbol should be ♚"
    assert board.piece_symbols['Q'] == '♕', "White queen symbol should be ♕"
    assert board.piece_symbols['q'] == '♛', "Black queen symbol should be ♛"
    print("✓ Unicode symbols test passed")
    
    print("\n🎉 All tests passed! The chess board is working correctly.")
    
    # Display the final position
    print("\nFinal test position:")
    board.display()

if __name__ == "__main__":
    test_basic_functionality() 