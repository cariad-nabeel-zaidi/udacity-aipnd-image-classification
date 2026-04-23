# -------------------------------
# PREDICT.PY - INFERENCE SCRIPT
# -------------------------------

import argparse
import torch
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import models

# -------------------------------
# 1. ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser(description="Predict flower name from image")

parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# -------------------------------
# 2. DEVICE
# -------------------------------
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# -------------------------------
# 3. LOAD CHECKPOINT
# -------------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

model = load_checkpoint(args.checkpoint)
model.to(device)
model.eval()

# -------------------------------
# 4. PROCESS IMAGE
# -------------------------------
def process_image(image_path):
    image = Image.open(image_path)

    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))

    np_image = np.array(image) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return np_image

# -------------------------------
# 5. PREDICT FUNCTION
# -------------------------------
def predict(image_path, model, topk=5):

    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        ps = F.softmax(output, dim=1)

        top_p, top_class = ps.topk(topk, dim=1)

    probs = top_p.cpu().numpy()[0]
    classes = top_class.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[c] for c in classes]

    return probs, classes

# -------------------------------
# 6. LOAD CATEGORY NAMES (OPTIONAL)
# -------------------------------
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

# -------------------------------
# 7. RUN PREDICTION
# -------------------------------
probs, classes = predict(args.image_path, model, args.top_k)

# Convert to flower names if mapping exists
if args.category_names:
    names = [cat_to_name[c] for c in classes]
else:
    names = classes

# -------------------------------
# 8. PRINT OUTPUT
# -------------------------------
print("\nTop Predictions:")
for i in range(len(names)):
    print(f"{names[i]} : {probs[i]:.3f}")