import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.small_model import SmallModel
from models.medium_model import MediumModel
from models.large_model import LargeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

def train(model, name, epochs=3):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            if name == "large":
                output = model(x, use_early_exit=False)
            else:
                output = model(x)

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        print(f"{name.upper()} Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        

    torch.save(model.state_dict(), f"weights/{name}.pth")


if __name__ == "__main__":
    import os
    os.makedirs("weights", exist_ok=True)

    train(SmallModel(), "small")
    train(MediumModel(), "medium")
    train(LargeModel(), "large")