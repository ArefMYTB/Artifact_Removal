import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from torchvision.models import ResNet50_Weights
from data_prepare import get_dataloaders

# Read the necessary parameters from the config file
with open("config.yml", 'r') as file:
    conf = yaml.safe_load(file)["classifier"]

Pretrain_PATH = conf["Pretrain_PATH"]

num_epochs = conf["num_epochs"]

label_mapping = {'Deformation': 0, 'Texture': 1, 'None': 2}

# Load the pre-trained ResNet model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the number of the input channels
weight = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    model.conv1.weight[:, :3] = weight
    model.conv1.weight[:, 3] = model.conv1.weight[:, 0]

# Modify the last layer to represent the 3 classes "deformation", "texture" and "none"
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

# summary(model, (4, 224, 224))

# Load and preprocess the dataset
train_loader, val_loader, test_loader = get_dataloaders()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training #######################################
for epoch in tqdm(range(num_epochs)):
    for batch in train_loader:
        distorted_images = batch['distorted']
        mask_images = batch['mask']

        labels = batch['label']
        labels = [label_mapping[label] for label in labels]
        labels = torch.tensor(labels)

        inputs = []
        for distorted_image, mask_image in zip(distorted_images, mask_images):
            inputs.append(torch.cat((distorted_image, mask_image), dim=0))
        inputs = torch.stack(inputs)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, Pretrain_PATH)

# Evaluation #####################################
# Set the model to evaluation mode
model.eval()

# Initialize evaluation variables
total_samples = 0
correct_predictions = 0

# Disable gradients for evaluation
with torch.no_grad():
    for batch in val_loader:
        distorted_images = batch['distorted']
        mask_images = batch['mask']

        labels = batch['label']
        labels = [label_mapping[label] for label in labels]
        labels = torch.tensor(labels)

        inputs = []
        for distorted_image, mask_image in zip(distorted_images, mask_images):
            inputs.append(torch.cat((distorted_image, mask_image), dim=0))
        inputs = torch.stack(inputs)

        # Forward pass and prediction
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)

        # Update evaluation variables
        total_samples += labels.size(0)
        correct_predictions += (predicted_labels == labels).sum().item()

# Calculate accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy}")
