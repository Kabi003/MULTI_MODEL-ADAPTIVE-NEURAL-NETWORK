import torch
import torch.nn.functional as F

from models.small_model import SmallModel
from models.medium_model import MediumModel
from models.large_model import LargeModel
from utils.resource_monitor import get_resource_score


class AdaptiveController:
    def __init__(self):
        self.small = SmallModel()
        self.medium = MediumModel()
        self.large = LargeModel()

    def load_weights(self):
        try:
            self.small.load_state_dict(torch.load("weights/small.pth"))
            self.medium.load_state_dict(torch.load("weights/medium.pth"))
            self.large.load_state_dict(torch.load("weights/large.pth"))
            print("✅ Models loaded")
        except:
            print("⚠️ Train models first")

    def get_confidence(self, output):
        probs = F.softmax(output, dim=1)
        return torch.max(probs).item()

    def select_model(self, x):
        resource_score = get_resource_score()

        # 🔥 Step 1: Resource-based selection
        if resource_score > 0.7:
            model = self.small
            mode = "SMALL"
            output = model(x)

        elif resource_score > 0.4:
            model = self.medium
            mode = "MEDIUM"
            output = model(x)

        else:
            model = self.large
            mode = "LARGE"
            output = model(x, use_early_exit=True)

        # 🔥 Step 2: Confidence-based escalation
        confidence = self.get_confidence(output)

        if confidence < 0.6 and mode != "LARGE":
            output = self.large(x, use_early_exit=False)
            mode = "ESCALATED_TO_LARGE"

        return output, mode, confidence