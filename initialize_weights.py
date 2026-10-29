import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from model_utils import build_model, CLASSES

def train_and_save():
    print("Initializing NapoVision MobileNetV2 model...")
    model = build_model(num_classes=len(CLASSES))
    
    # Preset image paths
    preset_data = [
        ("static/presets/supine.jpg", 0),  # Supine/Back
        ("static/presets/side.jpg", 1),    # Side
        ("static/presets/fetal.jpg", 2),   # Fetal
        ("static/presets/prone.jpg", 3)    # Prone/Stomach
    ]
    
    # Verify all files exist
    for path, _ in preset_data:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preset image: {path}")

    # Strong training augmentations
    aug_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_images = [(Image.open(p).convert("RGB"), label) for p, label in preset_data]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Fine-tuning classifier head on posture anchor samples...")
    model.train()
    for epoch in range(35):
        epoch_loss = 0.0
        # Create augmented batch of 32 images (8 of each class)
        batch_tensors = []
        batch_labels = []
        for _ in range(8):
            for img, label in raw_images:
                batch_tensors.append(aug_transform(img))
                batch_labels.append(label)

        inputs = torch.stack(batch_tensors)
        targets = torch.tensor(batch_labels, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/35] - Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in raw_images:
            inp = val_transform(img).unsqueeze(0)
            out = model(inp)
            pred = out.argmax(dim=1).item()
            conf = torch.softmax(out, dim=1)[0, pred].item()
            cls_name = CLASSES[pred]
            target_name = CLASSES[label]
            is_match = (pred == label)
            if is_match:
                correct += 1
            print(f"Validation Sample: Target={target_name} -> Predicted={cls_name} (Confidence: {conf*100:.2f}%)")

    print(f"Validation Accuracy: {correct}/{len(raw_images)} ({correct/len(raw_images)*100:.1f}%)")
    
    # Save checkpoint
    out_path = "napovision_model.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Model successfully saved to {out_path}")

if __name__ == "__main__":
    train_and_save()
