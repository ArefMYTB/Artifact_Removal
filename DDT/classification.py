import torch
import torch.nn as nn


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
        reduced_height = input_height // 4  # Assuming 2 pooling layers
        reduced_width = input_width // 4

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * reduced_height * reduced_width, 256),  # Calculate the reduced_height and reduced_width
            nn.ReLU(),
            nn.Dropout(0.5),  # Apply dropout for regularization
            nn.Linear(256, output_classes),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, distorted, mask, flawless):
        distorted_features = self.conv_layers(distorted)
        mask_features = self.conv_layers(mask)
        flawless_features = self.conv_layers(flawless)

        # Concatenate or sum the features from all inputs
        fused_features = torch.cat((distorted_features, mask_features, flawless_features), dim=1)

        fused_features = fused_features.view(fused_features.size(0), -1)  # Flatten

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
model = SimpleClassifier(input_channels, output_classes)
# model = MultiHeadClassifier(input_channels, output_classes)
