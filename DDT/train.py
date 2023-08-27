import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

from classification import SimpleClassifier, MultiHeadClassifier

# Instantiate the model
input_channels = 3  # Assuming RGB images
output_classes = 1  # Binary classification
model = SimpleClassifier(input_channels, output_classes)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load dataset using torchvision's ImageFolder
# Define the transformations for image preprocessing
transform = Resize((224, 224))
normalize = ToTensor()

train_dataset = ImageFolder(root='path/to/dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        distorted_images = inputs[:, 0:3, :, :]  # Adjust indexing based on dataset structure
        mask_images = inputs[:, 3:6, :, :]
        flawless_images = inputs[:, 6:9, :, :]

        outputs = model(distorted_images, mask_images, flawless_images)
        loss = criterion(outputs, labels.unsqueeze(1).float())  # BCELoss expects labels in (batch_size, 1) shape

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Save the trained model
torch.save(model, 'path/to/save/model.pth')
