import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

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

# Model
class ChessBoardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 8 * 8 * 13)  # for 256x256 input
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, 8, 8, 13)
        return x

def accuracy_per_square(pred, target):
    pred_labels = pred.argmax(dim=-1)
    correct = (pred_labels == target).float().sum()
    total = np.prod(target.shape)
    return correct / total

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = ChessBoardDataset('dataset/labels.csv', 'dataset/images', split='train', transform=transform)
    val_dataset = ChessBoardDataset('dataset/labels.csv', 'dataset/images', split='val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = ChessBoardNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # (batch, 8, 8, 13)
            loss = criterion(outputs.view(-1, 13), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                acc = accuracy_per_square(outputs.cpu(), labels.cpu())
                val_acc += acc.item()
        val_acc /= len(val_loader)
        print(f"  Validation per-square accuracy: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'chessboard_net.pth')
    print("Model saved as chessboard_net.pth")

if __name__ == '__main__':
    main() 