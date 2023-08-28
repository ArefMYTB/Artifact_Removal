import torch
import torchvision.transforms as transforms
from classification import DistortionClassifier
from PIL import Image


def load_model():
    # Load pre-trained model
    model = DistortionClassifier()
    model.load_state_dict(torch.load('path_to_model.pth'))
    model.eval()  # Set the model to evaluation mode
    return model


# get the distorted image and a corresponding mask
def ddt(image, mask):
    model = load_model()
    with torch.no_grad():
        output = model(image.unsqueeze(0), mask.unsqueeze(0))  # Add batch dimension
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_labels = ['Distorted', 'Texture', 'None']
    predicted_label = class_labels[predicted_class]

    print(f'Predicted class: {predicted_label}')


def main():
    # Load image and mask
    image = Image.open('image.png').convert('RGB')
    mask = Image.open('mask_1.png').convert('L')

    # Define transformation to be applied to the images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    mask = preprocess(mask)

    ddt(image, mask)


if __name__ == '__main__':
    main()

