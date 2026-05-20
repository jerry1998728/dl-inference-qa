"""
Train a CNN on MNIST using PyTorch and save the model weights to disk.

Architecture mirrors scripts/train_keras.py so the two trained models can be
compared in cross-framework tests later.

Run: python scripts/train_torch.py
Output: models/mnist_cnn_torch.pt
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Reproducibility ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# --- Config ---
MODEL_PATH = "models/mnist_cnn_torch.pt"
EPOCHS = 2
BATCH_SIZE = 512
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MnistCNN(nn.Module):
    """
    Same shape as the Keras model in train_keras.py:
        Conv(1->16, k=5, pad=same) -> ReLU -> MaxPool(2)
        Conv(16->36, k=5, pad=same) -> ReLU -> MaxPool(2)
        Flatten -> Dense(128) -> ReLU -> Dropout(0.2) -> Dense(10)
    Final softmax is applied inside CrossEntropyLoss during training; at inference
    we apply it explicitly so outputs match Keras (which has softmax in the model).
    """
    def __init__(self):
        super().__init__()
        # padding=2 with kernel=5 = "same" padding (output spatial dims unchanged)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(36 * 7 * 7, 128)   # after two 2x pools: 28 -> 14 -> 7
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))    # -> (N, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))    # -> (N, 36, 7, 7)
        x = x.flatten(1)                         # -> (N, 36*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                          # -> (N, 10) logits, NO softmax here
        return x


def get_loaders():
    """MNIST → tensors in [0, 1] with shape (N, 1, 28, 28)."""
    transform = transforms.Compose([
        transforms.ToTensor(),                   # converts PIL [0,255] -> tensor [0,1]
    ])
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Use first 5000 of train as validation (mirrors Keras script)
    valid_set = torch.utils.data.Subset(train_full, range(5000))
    train_set = torch.utils.data.Subset(train_full, range(5000, 60000))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, valid_loader, test_loader


def evaluate(model, loader):
    """Compute (avg_loss, accuracy) over a loader."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, total_correct / total


def main():
    print(f"PyTorch {torch.__version__}  |  device: {DEVICE}")
    train_loader, valid_loader, test_loader = get_loaders()

    model = MnistCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()                            # turn dropout ON
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()                # clear stale gradients
            logits = model(x)
            loss = loss_fn(logits, y)            # CrossEntropyLoss = softmax + NLL inside
            loss.backward()                      # compute gradients
            optimizer.step()                     # update weights

        val_loss, val_acc = evaluate(model, valid_loader)
        print(f"Epoch {epoch}/{EPOCHS}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model state_dict to {MODEL_PATH}")


if __name__ == "__main__":
    main()