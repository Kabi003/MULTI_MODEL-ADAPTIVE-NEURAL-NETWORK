import torch
from controller.adaptive_controller import AdaptiveController

# Initialize system
controller = AdaptiveController()
controller.load_weights()

# Simulated input
input_data = torch.randn(1, 1, 28, 28)

# Run adaptive inference
output, mode, confidence = controller.select_model(input_data)

print("\n🚀 Adaptive Inference Result")
print("Mode:", mode)
print("Confidence:", round(confidence, 4))
print("Prediction:", torch.argmax(output).item())