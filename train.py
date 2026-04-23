# -------------------------------
# TRAIN.PY - MODEL TRAINING SCRIPT
# -------------------------------

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# -------------------------------
# 1. ARGUMENTS (COMMAND LINE)
# -------------------------------
parser = argparse.ArgumentParser(description="Train a neural network")

parser.add_argument('data_dir', help='Dataset directory')
parser.add_argument('--save_dir', default='checkpoint.pth')
parser.add_argument('--arch', default='vgg16')
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# -------------------------------
# 2. DEVICE SETUP
# -------------------------------
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# -------------------------------
# 3. DATA LOADING
# -------------------------------
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
}

# -------------------------------
# 4. MODEL SELECTION
# -------------------------------
if args.arch == "vgg13":
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

# Freeze feature layers
for param in model.parameters():
    param.requires_grad = False

# -------------------------------
# 5. CLASSIFIER
# -------------------------------
model.classifier = nn.Sequential(
    nn.Linear(25088, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

model.to(device)

# -------------------------------
# 6. LOSS + OPTIMIZER
# -------------------------------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# -------------------------------
# 7. TRAINING LOOP
# -------------------------------
steps = 0

for epoch in range(args.epochs):
    running_loss = 0
    model.train()

    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------------------
    # VALIDATION
    # ---------------------------
    model.eval()
    accuracy = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)

            logps = model(images)
            val_loss += criterion(logps, labels).item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"\nEpoch {epoch+1}/{args.epochs}")
    print(f"Training Loss: {running_loss/len(dataloaders['train']):.3f}")
    print(f"Validation Loss: {val_loss/len(dataloaders['valid']):.3f}")
    print(f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# -------------------------------
# 8. SAVE CHECKPOINT
# -------------------------------
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
    'arch': args.arch,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
}

torch.save(checkpoint, args.save_dir)

print("\nModel saved successfully!")