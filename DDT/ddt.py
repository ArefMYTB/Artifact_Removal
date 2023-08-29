import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml

# Read the necessary parameters from the config file
with open("config.yml", 'r') as file:
    conf = yaml.safe_load(file)

Pretrain_PATH = conf["classifier"]["Pretrain_PATH"]
label_mapping = {'Deformation': 0, 'Texture': 1, 'None': 2}


def load_model():
    # Load pre-trained model
    model = torch.load(Pretrain_PATH)
    model.eval()  # Set the model to evaluation mode
    return model


# Use DDT Classifier to determine what kind of distortion we need to refine
def classify(model, img):
    with torch.no_grad():
        output = model(img)
        _, predicted_labels = torch.max(output, 1)

    class_labels = ['Distorted', 'Texture', 'None']
    predicted_label = class_labels[predicted_labels]

    print(f'Predicted class: {predicted_label}')
    return predicted_label, output


# necessary information for deform distortion
def deformation_info():
    print("Distorted function called")
    return []


# necessary information for texture distortion
def texture_info():
    print("Texture function called")
    return []


# necessary information for none distortion
def none_info():
    print("None function called")
    return []


# get the distorted image and a corresponding mask
def ddt(img):
    model = load_model()
    predicted_label, probabilities = classify(model, img)

    if predicted_label == label_mapping['None']:
        none_information = none_info()
        return none_information, probabilities
    else:
        deformation_information = deformation_info()
        texture_information = texture_info()
        return [deformation_information, texture_information], probabilities


def main():
    # Load image and mask
    image = Image.open('distorted.png')
    mask = Image.open('mask_1.png').convert('L')

    input_width = conf["data"]["input_width"]
    input_height = conf["data"]["input_height"]

    # Define transformation to be applied to the images
    preprocess = transforms.Compose([
        transforms.Resize((input_width, input_height)),
        transforms.ToTensor()
    ])

    image = preprocess(image)
    mask = preprocess(mask)

    # concat image & mask
    fused_features = torch.cat((image, mask), dim=0)
    fused_features = torch.stack([fused_features])

    ddt(fused_features)


if __name__ == '__main__':
    main()
