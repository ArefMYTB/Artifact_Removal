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


def classify(model, img):
    with torch.no_grad():
        # output = model(image.unsqueeze(0), mask.unsqueeze(0))  # Add batch dimension
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_labels = ['Distorted', 'Texture', 'None']
    predicted_label = class_labels[predicted_class]

    print(f'Predicted class: {predicted_label}')
    return predicted_label, probabilities


def distorted_info():
    print("Distorted function called")
    return []


def texture_info():
    print("Texture function called")
    return []


def none_info():
    print("None function called")
    return []


# get the distorted image and a corresponding mask
def ddt(img):
    model = load_model()
    predicted_label, probabilities = classify(model, img)

    if predicted_label == ['None']:
        none_information = none_info()
        return none_information, probabilities
    else:
        distorted_information = distorted_info()
        texture_information = texture_info()
        return [distorted_information, texture_information], probabilities


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

    # concat image & mask
    fused_features = torch.cat((image, mask), dim=1)
    # fused_features = fused_features.view(fused_features.size(0), -1)  # Flatten

    ddt(fused_features)


if __name__ == '__main__':
    main()
