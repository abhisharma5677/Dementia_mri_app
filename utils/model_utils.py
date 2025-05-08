import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import gdown

# Define model download URL and local path
MODEL_URL = "https://drive.google.com/uc?id=1-P56CSh_T7V0urBiswars92Cs2hSpuHd"
MODEL_PATH = os.path.join("app", "resnet50_dementia.pth")

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("app", exist_ok=True)
        print("📥 Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded.")

def load_model():
    download_model_if_needed()
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model):
    transform = ResNet50_Weights.DEFAULT.transforms()
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]
