import os
import timm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from helper_functions.data_processor import predict_single_image, parse_json, MultiLabelAffordanceDataset, \
    multi_label_accuracy

torch.manual_seed(0)

data_dir = "data/RGBD affordance dataset/JSON format"
image_data_dir = 'data/RGBD affordance dataset/RGB images'

# Create an empty dictionary to store image-label mappings for all images
image_label_mapping = {}
for fold in os.listdir(data_dir):
    print(f"------'{fold}'--------- ")
    class_path = os.path.join(data_dir, fold)
    for class_name in os.listdir(class_path):
        print(f"++++++{class_name}+++++++")
        json_path = os.path.join(class_path, class_name)
        # json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]

        # Iterate over each JSON file in the directory
        for json_filename in os.listdir(json_path):
            if json_filename.endswith('.json'):
                json_file_path = os.path.join(json_path, json_filename)
                file_name, labels = parse_json(json_file_path)
                image_name = os.path.join(image_data_dir, file_name + '.png')
                # Update the dictionary with the new image-label mapping
                image_label_mapping[image_name] = labels

print("finalising data preparation")

# Define the transform for your images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Example resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

all_possible_labels = set()

for labels in image_label_mapping.values():
    all_possible_labels.update(labels)

print(all_possible_labels)

all_possible_labels_list = list(all_possible_labels)

num_labels = len(all_possible_labels)  # Number of unique labels in your dataset

dataset = MultiLabelAffordanceDataset(data_dir, image_label_mapping, all_possible_labels)

# Split the dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

trainLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validationLoader = DataLoader(val_dataset, batch_size=32, shuffle=False)
testLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading model")
# Load the model
model_name = "swin_base_patch4_window7_224"
model = timm.create_model(model_name, pretrained=True)
model = model.eval()

print(model)
print(model.head.in_features)
# adjusting model head to affordance labels
model.head.fc = torch.nn.Linear(model.head.in_features, num_labels)

model = model.to(device)

print("defining optimiser")
# Loss and Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# Training Loop
num_epochs = 10
train_loss_list = []
train_acc_list = []
val_acc_list = []
val_loss_list = []

print("start training model")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    print(epoch)
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())

        acc = multi_label_accuracy(outputs, labels)
        total_train_acc += acc

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(trainLoader)
    average_train_acc = total_train_acc / len(trainLoader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_train_loss:.4f}, Training Accuracy: {average_train_acc:.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_train_loss:.4f}")

    train_loss_list.append(average_train_loss)
    train_acc_list.append(average_train_acc)

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
        for images, labels in validationLoader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = multi_label_accuracy(outputs, labels)

            total_val_acc += acc
            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(validationLoader)
    average_val_acc = total_val_acc / len(validationLoader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {average_val_loss:.4f}")
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {average_val_acc:.4f}")
    val_loss_list.append(average_val_loss)
    val_acc_list.append(average_val_acc)

# Plotting training and validation loss
import matplotlib.pyplot as plt

# Plotting training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Training Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Test phase (after all epochs are finished)
model.eval()
total_test_loss = 0
total_test_acc = 0
with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = multi_label_accuracy(outputs, labels)
        total_test_loss += loss.item()
        total_test_acc += acc

average_test_loss = total_test_loss / len(testLoader)
average_test_acc = total_test_acc / len(testLoader)
print(f"Test Loss: {average_test_loss:.4f}")
print(f"Test Accuracy: {average_test_acc:.4f}")

test_dataset_index = 1500
print(test_dataset.dataset.image_files[test_dataset_index])

image_path = test_dataset.dataset.image_files[test_dataset_index]
print(image_path)
model.to("cuda")
predictions = predict_single_image(image_path, model, list(all_possible_labels))
print(predictions)

print("saving model and other data")

torch.save(model.state_dict(),
           f"models/model_swin_base_patch4_window7_224_affordance_state_dict_epoch_{num_epochs}.pth")
torch.save(model, f"models/model_swin_base_patch4_window7_224_affordance__{num_epochs}.pth")

import pickle

with open(f'data/test_dataset_swin_base_patch4_window7_224_torchhub_epoch{num_epochs}.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

with open(f'data/train_dataset_swin_base_patch4_window7_224_torchub_epoch{num_epochs}.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open(f'data/val_dataset_swin_base_patch4_window7_224_torchhub_epoch{num_epochs}.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)

# print(model)

with open(f"data/all_possible_labels_list_swin_base_patch4_window7_224_epoch{num_epochs}.pkl", 'wb') as f:
    pickle.dump(all_possible_labels_list, f)

import shutil
import os

source_dir = image_data_dir
print(source_dir)
dest_dir = "data/" + f"Test Images-swin_base_patch4_window7_224'_{num_epochs}'"

for test_image_index in sorted(test_dataset.indices):

    print(test_image_index)
    file_path = os.path.normpath(test_dataset.dataset.image_files[test_image_index])
    print(file_path)
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Copy all files from source directory to destination directory
    if os.path.isfile(file_path):
        shutil.copy(file_path, dest_dir)

print("done")

print(test_dataset.dataset.all_possible_labels)
print(all_possible_labels_list)
