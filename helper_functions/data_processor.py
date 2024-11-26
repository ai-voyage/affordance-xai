import json
import os

import timm
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from transformers import ViTForImageClassification, AdamW, DeiTForImageClassification
import torch.nn.functional as F


def parse_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # image_path = data['imagePath']
    filename = filename = json_file.split('.json')[0].split('/')[-1]
    labels = [shape['label'] for shape in data['shapes']]

    # return image_path, labels
    return filename, labels


def extract_labels_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        # Extract labels from the shapes
        labels = [shape['label'] for shape in data['shapes']]
        return labels

class MultiLabelAffordanceDataset(Dataset):
    def __init__(self, images_dir, image_label_mapping, all_possible_labels):
        self.images_dir = images_dir
        self.image_files = list(image_label_mapping.keys())
        self.labels_list = list(image_label_mapping.values())
        self.all_possible_labels = all_possible_labels

        # Define the transform for your images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Example resize
            transforms.ToTensor(),  # This will automatically scale pixel values to [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name)

        # Apply transformations
        image = self.transform(image)

        # Convert the list of labels for this image into a binary vector
        binary_vector = [1 if label in self.labels_list[idx] else 0 for label in self.all_possible_labels]

        return image, torch.tensor(binary_vector, dtype=torch.float32)  # float32 is suitable for BCEWithLogitsLoss

def multi_label_accuracy(logits, labels, threshold=0.5):
    """Compute accuracy for multi-label classification"""
    probs = F.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == labels).float().sum(dim=1)  # number of correct labels per sample
    acc = (correct == labels.size(1)).float().mean()  # check if all labels per sample are correct
    return acc.item()


def predict_single_image(image_path, model, label_list, threshold=0.5, device="cuda"):
    # Predict affordances for a single image.


    # Load image
    image = Image.open(image_path).convert("RGB")

    # Same preprocessing as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Pass image through the model
    model.eval()
    with torch.no_grad():
        logits = model(image)

    # Convert logits to probabilities
    probs = F.sigmoid(logits).squeeze().cpu().numpy()

    # Get binary predictions
    predicted_labels = (probs > threshold)

    # Convert predictions to label strings
    predictions = {label_list[i]: probs[i] for i in range(len(label_list)) if predicted_labels[i]}

    return predictions

