import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class DistortionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        distorted_images = []
        flawless_images = []
        mask_images = []
        original_images = []
        print(folder_path)
        for file in os.listdir(folder_path):
            if file.startswith("distorted"):
                distorted_img = Image.open(os.path.join(folder_path, file))
                distorted_images.append(distorted_img)
            if file.startswith("flawless"):
                flawless_img = Image.open(os.path.join(folder_path, file))
                flawless_images.append(flawless_img)
            elif file.startswith("mask"):
                mask_img = Image.open(os.path.join(folder_path, file))
                mask_images.append(mask_img)
            elif file.startswith("original"):
                original_img = Image.open(os.path.join(folder_path, file))
                original_images.append(original_img)

        sample = {
            'distorted': distorted_images,
            'flawless': flawless_images,
            'mask': mask_images,
            'original': original_images
        }

        if self.transform:
            sample['distorted'] = [self.transform(img) for img in sample['distorted']]
            sample['flawless'] = [self.transform(img) for img in sample['flawless']]
            sample['mask'] = [self.transform(img) for img in sample['mask']]
            sample['original'] = [self.transform(img) for img in sample['original']]

        return sample

def main():
    # Define transformation to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((748, 512)),
        transforms.ToTensor(),
    ])

    # Define paths to deformation and texture folders
    deformation_folder = 'Dataset/DHI/Deformation'
    texture_folder = 'Dataset/DHI/Texture'

    # Create dataset instances for deformation and texture
    deformation_dataset = DistortionDataset(root_dir=deformation_folder, transform=data_transform)
    texture_dataset = DistortionDataset(root_dir=texture_folder, transform=data_transform)

    # Create data loaders
    batch_size = 32
    deformation_loader = DataLoader(deformation_dataset, batch_size=batch_size, shuffle=True)
    texture_loader = DataLoader(texture_dataset, batch_size=batch_size, shuffle=True)

    # batch = next(iter(deformation_loader))
    # # print(batch['mask'][0].shape)
    # batch_tensor = batch['distorted'][0]
    #
    # Convert tensor to a PIL image
    transform = transforms.ToPILImage()
    # # pil_image = transform(transposed_image_tensor)
    #
    # for i in range(batch_tensor.size(0)):
    #     image_tensor = batch_tensor[i]
    #     pil_image = transform(image_tensor)
    #     pil_image.show()

    for batch in tqdm(deformation_loader):
        image_tensor = batch['distorted'][0][0]
        pil_image = transform(image_tensor)
        pil_image.show()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     for batch_idx, (deformation_images, texture_images) in enumerate(zip(deformation_loader, texture_loader)):
    #         deformation_images = torch.stack(deformation_images).to(device)
    #         texture_images = torch.stack(texture_images).to(device)
    #
    #         # Your training/validation steps here
    #         # Example: model.forward(deformation_images, texture_images)


if __name__ == '__main__':
    main()
