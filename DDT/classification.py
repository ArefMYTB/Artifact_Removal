import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image


class SimpleClassifier(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(SimpleClassifier, self).__init__()

        # Shared convolutional layers for each input type
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more convolutional layers as needed
        )

        # Calculate the dimensions after the convolutional layers
        flattened_size = 4325376

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),  # Calculate the reduced_height and reduced_width
            nn.ReLU(),
            nn.Dropout(0.5),  # Apply dropout for regularization
            nn.Linear(256, output_classes),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, distorted, mask, flawless):
        distorted_features = self.conv_layers(distorted)
        mask_features = self.conv_layers(mask)
        flawless_features = self.conv_layers(flawless)

        resized_flawless_features = F.interpolate(mask_features,
                                                  size=(distorted_features.size(2),
                                                        distorted_features.size(3)),
                                                  mode='bilinear', align_corners=False)

        # Concatenate or sum the features from all inputs
        fused_features = torch.cat((distorted_features, mask_features, resized_flawless_features), dim=1)

        fused_features = fused_features.view(fused_features.size(0), -1)  # Flatten

        # batch_size, flattened_size = fused_features.size()
        # print(flattened_size)

        output = self.fc_layers(fused_features)
        return output


class MultiHeadClassifier(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(MultiHeadClassifier, self).__init__()

        # Convolutional layers for the distorted image input
        self.conv_distorted = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more convolutional layers as needed
        )

        # Convolutional layers for the mask input
        self.conv_mask = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more convolutional layers as needed
        )

        # Convolutional layers for the flawless image input
        self.conv_flawless = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more convolutional layers as needed
        )

        # Calculate the dimensions after the convolutional layers
        reduced_height = input_height // 4  # Assuming 2 pooling layers
        reduced_width = input_width // 4

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * reduced_height * reduced_width * 3, 256),  # Multiply by 3 for 3 input types
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_classes),
            nn.Sigmoid()
        )

    def forward(self, distorted, mask, flawless):
        distorted_features = self.conv_distorted(distorted)
        mask_features = self.conv_mask(mask)
        flawless_features = self.conv_flawless(flawless)

        # Concatenate features from all input types
        fused_features = torch.cat((distorted_features, mask_features, flawless_features), dim=1)

        fused_features = fused_features.view(fused_features.size(0), -1)  # Flatten

        output = self.fc_layers(fused_features)
        return output


# Instantiate the model
input_channels = 3  # RGB images
output_classes = 1  # Binary classification
input_height = 224  # input image size
input_width = 224

distorted_image_path = "distorted.png"
distorted_image = Image.open(distorted_image_path)
# print(distorted_image.size)
mask_path = "mask_1.png"
mask_image = Image.open(mask_path)
flawless_image_path = "flawless.jpg"
flawless_image = Image.open(flawless_image_path)
# print(flawless_image.size)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
])
distorted_image_tensor = transform(distorted_image).unsqueeze(0)  # Add a batch dimension
mask_image_tensor = transform(mask_image).unsqueeze(0)
flawless_image_tensor = transform(flawless_image).unsqueeze(0)


model = SimpleClassifier(input_channels, output_classes)
res = model.forward(distorted_image_tensor, mask_image_tensor, flawless_image_tensor)
print(res)
# model = MultiHeadClassifier(input_channels, output_classes)
