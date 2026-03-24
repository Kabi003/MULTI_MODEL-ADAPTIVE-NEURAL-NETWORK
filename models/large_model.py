import torch.nn as nn
import torch.nn.functional as F

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        self.early_exit = nn.Linear(128, 10)

    def forward(self, x, use_early_exit=False):
        x = x.view(-1, 28*28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if use_early_exit:
            return self.early_exit(x)

        x = F.relu(self.fc3(x))
        return self.output(x)