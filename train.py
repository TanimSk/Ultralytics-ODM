import torch
from ultralytics import YOLO, checks, hub


API_KEY = ""
MODEL_URL = ""

# Set device to MPS (Metal Performance Shaders) for Macs, CUDA for Nvidia GPUs, or CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checks()
hub.login(api_key=API_KEY)


# Load YOLO model
model = YOLO(MODEL_URL )

# Train using MPS
results = model.train(device=device)
