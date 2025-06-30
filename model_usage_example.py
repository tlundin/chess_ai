import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the model architecture from the training script
from train_chessboard_recognizer_diverse import ChessBoardNet, FEN_TO_IDX, IDX_TO_FEN

def load_trained_model(model_path='best_chessboard_net_diverse.pth'):
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
    print(f"📊 Dataset: {checkpoint.get('dataset', 'unknown')}")
    
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

def test_single_image(image_path, model, device):
    """Test a single image and show results"""
    print(f"\n📸 Processing image: {image_path}")
    print("=" * 60)
    
    # Preprocess the image
    image_tensor, original_image = preprocess_image(image_path)
    
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
    save_path = f"prediction_result_{os.path.basename(image_path).split('.')[0]}.png"
    visualize_predictions(original_image, board, probabilities, save_path)
    
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

def main():
    """Main function to demonstrate model usage"""
    print("🎮 Chess Board Recognition Demo - Improved Model")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'best_chessboard_net_diverse.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first using train_chessboard_recognizer_diverse.py")
        return
    
    # Load the trained model
    model, device = load_trained_model(model_path)
    
    # Check if test_sample directory exists
    test_dir = 'test_sample'
    if not os.path.exists(test_dir):
        print(f"❌ Test directory '{test_dir}' not found!")
        return
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print(f"❌ No images found in '{test_dir}' directory!")
        return
    
    print(f"\n📁 Found {len(test_images)} test images:")
    for i, img in enumerate(test_images, 1):
        print(f"   {i}. {img}")
    
    print(f"\n🚀 Starting testing with improved model...")
    print("Press Enter after each image to continue to the next one.")
    
    for i, image_filename in enumerate(test_images, 1):
        image_path = os.path.join(test_dir, image_filename)
        
        print(f"\n{'='*70}")
        print(f"📸 Testing Image {i}/{len(test_images)}: {image_filename}")
        print(f"{'='*70}")
        
        # Test the image
        test_single_image(image_path, model, device)
        
        # Wait for user input before continuing
        if i < len(test_images):
            input(f"\n⏸️  Press Enter to continue to the next image ({i+1}/{len(test_images)})...")
        else:
            print(f"\n🎉 All {len(test_images)} images tested!")
    
    print(f"\n✅ Testing completed! Check the generated prediction images for results.")

if __name__ == "__main__":
    main() 