import torch.nn as nn
import torch.nn.functional as F

class MediumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.exit = nn.Linear(64, 10)  # early exit

    def forward(self, x, early_exit=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if early_exit:
            return self.exit(x)

        return self.exit(x)