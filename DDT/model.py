import yaml
from tqdm import tqdm
# from torchvision.models import ResNet50_Weights
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchvision import transforms
import timm  # This is for Vision Transformer (ViT)
import cv2
from PIL import Image
import numpy as np

class CoefficientGenerator(nn.Module):
    def __init__(self, resolution, num_output=6):
        super(CoefficientGenerator, self).__init__()
        
        # Initialize a pre-trained ResNet-100 model
        self.resnet100 = models.resnet101(pretrained=True)
        self.fc_in_features = self.resnet100.fc.in_features
        
        self.resolution = resolution
        
        # Initialize Vision Transformer (ViT)
        self.vit = timm.create_model('resnet101', pretrained=True)
        self.vit = torch.nn.Sequential(*list(self.vit.children())[:-1])

        # Define attention mechanism for combining image and mask
        self.attention = nn.MultiheadAttention(embed_dim=self.fc_in_features, num_heads=8)

        # Output layer
        self.fc = nn.Linear(self.fc_in_features, num_output)

        # Initialize resnet_layers for feature extraction
        self.resnet_layers = nn.Sequential(*list(self.resnet100.children())[:-1])  # Remove the last two layers
        
        # TODO get this from config
        freeze_layer_number = 70
        # Freeze the layers up to freeze_layer_number
        self.freeze_layers(freeze_layer_number)

    def freeze_layers(self, num):
        for param in self.resnet_layers[:num].parameters():
            param.requires_grad = False

    def forward(self, image, mask):
        image = torch.stack(image)
        mask = torch.stack(mask)

        # Extract features from the ResNet model
        image_features = self.resnet_layers(image).view(image.shape[0], self.fc_in_features)

        # Expand the mask to match the number of channels in the image
        mask = mask.expand_as(image)
        
        with torch.no_grad():
          mask_features = self.vit(mask)

        # Apply attention mechanism to combine image and mask features
        combined_features, _ = self.attention(image_features, mask_features, mask_features)

        output = self.fc(combined_features.view(combined_features.size(0), -1))  # Flatten combined_features

        return output.tolist() # torch.sigmoid(output).tolist()

if __name__ == "__main__":
  
    generator = CoefficientGenerator(num_output=3)
    
    # image = torch.randn(1, 3, 512, 512)
    # mask = torch.randn(1, 1, 512, 512)  
    image_path = 'distorted.jpg'
    mask_path = 'mask.jpg'

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0) 

    mask = Image.open(mask_path)
    mask = mask.convert('L') 
    mask = transform(mask)
    mask = mask.unsqueeze(0)  

    coefficients = generator(image, mask)
    print("Generated Coefficients:", coefficients)
