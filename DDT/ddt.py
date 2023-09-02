import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import yaml
from controlnet_aux import OpenposeDetector, HEDdetector

# Read the necessary parameters from the config file
with open("config.yml", 'r') as file:
    conf = yaml.safe_load(file)

Pretrain_PATH = conf["model"]["Pretrain_PATH"]
label_mapping = {'HED': 0, 'Pose': 1, 'Texture': 2, 'Reference': 3}


def load_model():
    # Load pre-trained model
    model = torch.load(Pretrain_PATH)
    model.eval()  # Set the model to evaluation mode
    return model


# Use DDT Classifier
def classify(model, img):
    with torch.no_grad():
        output = model(img)
        _, predicted_labels = torch.max(output, 1)

    class_labels = ['Distorted', 'Pose', 'Texture', 'Reference']
    predicted_label = class_labels[predicted_labels]

    print(f'Predicted class: {predicted_label}')
    return predicted_label, output


def hed_condition(img):
    print("hed_condition called")

    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    hed_image = hed(img)

    return hed_image


def pose_condition(img):
    print("pose_condition called")

    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    pose_real_image = img

    pose_image = openpose(pose_real_image)
    pose_real_image = pose_real_image.resize(pose_image.size)

    pose_mask = np.zeros_like(np.array(pose_image))
    pose_mask[250:700, :, :] = 255
    pose_mask = Image.fromarray(pose_mask)

    return pose_image


def texture_condition():
    print("texture_condition called")
    return []


def reference_condition(reference_img):
    print("reference_condition called")
    return []


# get the distorted image and a corresponding mask
def ddt(img, reference_img):
    # model = load_model()
    # predicted_label, probabilities = classify(model, img)

    # if predicted_label == label_mapping['None']:
    #     none_information = none_info()
    #     return none_information, probabilities
    # else:
    #     deformation_information = deformation_info()
    #     texture_information = texture_info()
    #     return [deformation_information, texture_information], probabilities

    # TODO there's a problem with condtions loading
    return [[], [], [], []]
    # return [hed_condition(img), pose_condition(img), texture_condition(), reference_condition(reference_img)]


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
