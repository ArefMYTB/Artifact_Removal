import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml
from torchvision.models import ResNet50_Weights
from torchsummary import summary
from data_prepare import get_dataloaders

# Read the necessary parameters from the config file
with open("config.yml", 'r') as file:
    conf = yaml.safe_load(file)["classifier"]

num_epochs = conf["num_epochs"]

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

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # TODO: the following 3 lines must change!
        distorted_images = batch['distorted']
        mask_images = batch['mask']
        labels = batch['label']
        inputs = torch.cat((distorted_images, mask_images), dim=1)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
# Similar steps for validation and testing data
