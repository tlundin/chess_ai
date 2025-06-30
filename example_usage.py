#!/usr/bin/env python3
"""
Example usage of the ChessBoard class with graphical interface.
This script demonstrates how to use the chess board with both console and GUI display.
"""

from chess_board import ChessBoard
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the model architecture from the training script
from train_chessboard_recognizer_optimized import ChessBoardNet, FEN_TO_IDX, IDX_TO_FEN

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

def load_trained_model(model_path='best_chessboard_net.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChessBoardNet().to(device)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"🏆 Best validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"📅 Trained for {checkpoint['epoch'] + 1} epochs")
    
    return model, device

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess an image for the model"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor, image

def predict_chess_board(model, device, image_tensor):
    """Predict chess piece positions from an image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Get predictions for each square
        predictions = outputs.argmax(dim=-1)  # Shape: (1, 8, 8)
        probabilities = torch.softmax(outputs, dim=-1)  # Shape: (1, 8, 8, 13)
        
        return predictions.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def board_to_fen(board):
    """Convert the predicted board to FEN notation"""
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        
        for piece_idx in row:
            piece = IDX_TO_FEN[piece_idx]
            if piece == ' ':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    return '/'.join(fen_rows)

def visualize_predictions(original_image, board, probabilities, save_path=None):
    """Visualize the predictions on the original image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    ax1.imshow(original_image)
    ax1.set_title('Original Chess Board Image')
    ax1.axis('off')
    
    # Predicted board
    piece_symbols = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
        ' ': ' '
    }
    
    # Create a visual representation of the board
    board_visual = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            piece_idx = board[i, j]
            piece = IDX_TO_FEN[piece_idx]
            
            # Color squares (alternating pattern)
            if (i + j) % 2 == 0:
                board_visual[i, j] = [240, 217, 181]  # Light square
            else:
                board_visual[i, j] = [181, 136, 99]   # Dark square
    
    ax2.imshow(board_visual)
    ax2.set_title('Predicted Chess Board')
    
    # Add piece symbols
    for i in range(8):
        for j in range(8):
            piece_idx = board[i, j]
            piece = IDX_TO_FEN[piece_idx]
            symbol = piece_symbols[piece]
            
            if piece != ' ':
                color = 'white' if piece.isupper() else 'black'
                ax2.text(j, i, symbol, ha='center', va='center', 
                        fontsize=20, color=color, weight='bold')
    
    # Add grid
    ax2.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    plt.show()

def analyze_confidence(probabilities, board):
    """Analyze the confidence of predictions"""
    print("\n🎯 Prediction Confidence Analysis:")
    print("=" * 50)
    
    # Get confidence for each square
    confidences = np.max(probabilities, axis=-1)
    
    # Find squares with low confidence
    low_confidence_threshold = 0.8
    low_conf_squares = np.where(confidences < low_confidence_threshold)
    
    if len(low_conf_squares[0]) > 0:
        print(f"⚠️  Squares with low confidence (<{low_confidence_threshold}):")
        for i, j in zip(low_conf_squares[0], low_conf_squares[1]):
            piece = IDX_TO_FEN[board[i, j]]
            confidence = confidences[i, j]
            print(f"   Position ({i},{j}): {piece} (confidence: {confidence:.3f})")
    else:
        print("✅ All predictions have high confidence!")
    
    print(f"📊 Average confidence: {np.mean(confidences):.3f}")
    print(f"📊 Minimum confidence: {np.min(confidences):.3f}")
    print(f"📊 Maximum confidence: {np.max(confidences):.3f}")

def main():
    """Main function to demonstrate model usage"""
    print("🎮 Chess Board Recognition Demo")
    print("=" * 40)
    
    # Check if model exists
    model_path = 'best_chessboard_net.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first using train_chessboard_recognizer_optimized.py")
        return
    
    # Load the trained model
    model, device = load_trained_model(model_path)
    
    # Example: Use a test image from the dataset
    test_image_path = 'dataset/images/chess_board_0.png'  # Adjust path as needed
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image '{test_image_path}' not found!")
        print("Please provide a path to a chess board image")
        return
    
    print(f"\n📸 Processing image: {test_image_path}")
    
    # Preprocess the image
    image_tensor, original_image = preprocess_image(test_image_path)
    
    # Make prediction
    print("🔮 Making prediction...")
    board, probabilities = predict_chess_board(model, device, image_tensor)
    
    # Convert to FEN notation
    fen = board_to_fen(board)
    print(f"\n📋 Predicted FEN: {fen}")
    
    # Analyze confidence
    analyze_confidence(probabilities, board)
    
    # Visualize results
    print("\n🎨 Creating visualization...")
    visualize_predictions(original_image, board, probabilities, 'prediction_result.png')
    
    # Print board in a readable format
    print("\n📊 Predicted Board Layout:")
    print("  a b c d e f g h")
    print("  ---------------")
    for i, row in enumerate(board):
        row_str = f"{8-i} "
        for piece_idx in row:
            piece = IDX_TO_FEN[piece_idx]
            row_str += piece + " "
        row_str += f"{8-i}"
        print(row_str)
    print("  ---------------")
    print("  a b c d e f g h")

if __name__ == "__main__":
    main() 