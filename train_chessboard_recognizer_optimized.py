import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import time
import psutil
from tqdm import tqdm
import sys

# Enable cuDNN optimizations for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Map FEN chars to class indices
FEN_TO_IDX = {
    'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
    'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11,
    ' ': 12  # empty
}
IDX_TO_FEN = {v: k for k, v in FEN_TO_IDX.items()}

# Dataset class
class ChessBoardDataset(Dataset):
    def __init__(self, csv_path, images_dir, split='train', transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.images_dir, row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = fen_to_board(row['pgn'])  # shape (8,8)
        return img, torch.tensor(label)

def fen_to_board(fen):
    board = []
    for row in fen.split('/'):
        row_arr = []
        for c in row:
            if c.isdigit():
                row_arr.extend([' '] * int(c))
            else:
                row_arr.append(c)
        board.append([FEN_TO_IDX[x] for x in row_arr])
    return np.array(board, dtype=np.int64)  # shape (8,8)

# Improved Model with better architecture
class ChessBoardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Adaptive pooling to get 8x8 output
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 13, 1),  # 1x1 conv to get 13 classes per square
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.permute(0, 2, 3, 1)  # (batch, 8, 8, 13)
        return x

def accuracy_per_square(pred, target):
    pred_labels = pred.argmax(dim=-1)
    correct = (pred_labels == target).float().sum()
    total = np.prod(target.shape)
    return correct / total

def print_memory_usage():
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    
    memory = psutil.virtual_memory()
    return f"RAM: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)"

def find_optimal_batch_size(model, device, sample_input, max_batch_size=1024):
    """Find the optimal batch size that fits in GPU memory"""
    print("🔍 Finding optimal batch size...")
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    available_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3 if torch.cuda.is_available() else None
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            test_input = sample_input.repeat(batch_size, 1, 1, 1).to(device)
            test_output = model(test_input)
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  Batch size {batch_size}: {allocated:.2f}GB allocated, {cached:.2f}GB cached (of {available_mem:.2f}GB total)")
            del test_input, test_output
            torch.cuda.empty_cache()
            if allocated > 0.8 * available_mem:
                optimal_batch_size = batch_size // 2
                print(f"✅ Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_batch_size = batch_size // 2
                print(f"✅ Optimal batch size: {optimal_batch_size} (OOM at {batch_size})")
                return optimal_batch_size
            else:
                raise e
    print("⚠️  Could not find optimal batch size, using 64 as fallback.")
    return 64  # Default fallback

def main():
    print("🚀 Starting Optimized Chess Board Training")
    print("=" * 60)
    
    # Check hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Hyperparameters
    num_epochs = 10  # Reduced for testing
    lr = 1e-3
    
    # Data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ChessBoardDataset('dataset/labels.csv', 'dataset/images', split='train', transform=train_transform)
    val_dataset = ChessBoardDataset('dataset/labels.csv', 'dataset/images', split='val', transform=val_transform)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create model and find optimal batch size
    print("🧠 Creating model...")
    model = ChessBoardNet().to(device)
    
    # Test with a sample to find optimal batch size
    sample_img, _ = train_dataset[0]
    sample_input = sample_img.unsqueeze(0).to(device)
    optimal_batch_size = find_optimal_batch_size(model, device, sample_input)
    
    # Create data loaders with optimizations
    print(f"⚡ Creating data loaders with batch size: {optimal_batch_size}")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=optimal_batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=optimal_batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with monitoring
    best_val_acc = 0.0
    total_start_time = time.time()
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, ncols=100)
        
        for batch_idx, (imgs, labels) in enumerate(train_pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs.reshape(-1, 13), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred_labels = outputs.argmax(dim=-1)
            train_correct += (pred_labels == labels).float().sum().item()
            train_total += labels.numel()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_acc = train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'Mem': print_memory_usage().split(': ')[1].split(',')[0]
            })
        
        train_acc = train_correct / train_total
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_acc = 0
        val_loss = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(imgs)
                loss = criterion(outputs.reshape(-1, 13), labels.reshape(-1))
                val_loss += loss.item()
                
                acc = accuracy_per_square(outputs.cpu(), labels.cpu())
                val_acc += acc.item()
                
                # Update validation progress bar
                avg_val_loss = val_loss / (val_pbar.n + 1)
                current_val_acc = val_acc / (val_pbar.n + 1)
                val_pbar.set_postfix({
                    'Loss': f'{avg_val_loss:.4f}',
                    'Acc': f'{current_val_acc:.4f}'
                })
        
        val_acc /= len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_chessboard_net.pth')
            print(f"💾 New best model saved! Val Acc: {val_acc:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"   ⏱️  Time: {epoch_time:.2f}s")
        print(f"   📊 Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   📊 Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"   📚 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   💾 {print_memory_usage()}")
        print("-" * 60)
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Model saved as: best_chessboard_net.pth")

if __name__ == '__main__':
    main() 