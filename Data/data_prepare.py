import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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
        mask_images = []
        original_images = []

        for file in os.listdir(folder_path):
            if file.startswith("distorted"):
                distorted_img = Image.open(os.path.join(folder_path, file))
                distorted_images.append(distorted_img)
            elif file.startswith("mask"):
                mask_img = Image.open(os.path.join(folder_path, file))
                mask_images.append(mask_img)
            elif file.startswith("original"):
                original_img = Image.open(os.path.join(folder_path, file))
                original_images.append(original_img)

        sample = {
            'distorted': distorted_images,
            'mask': mask_images,
            'original': original_images
        }

        if self.transform:
            sample['distorted'] = [self.transform(img) for img in sample['distorted']]
            sample['mask'] = [self.transform(img) for img in sample['mask']]
            sample['original'] = [self.transform(img) for img in sample['original']]

        return sample

def main():
    # Define transformation to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Define paths to deformation and texture folders
    deformation_folder = 'path_to_deformation_folder'
    texture_folder = 'path_to_texture_folder'

    # Create dataset instances for deformation and texture
    deformation_dataset = DistortionDataset(root_dir=deformation_folder, transform=data_transform)
    texture_dataset = DistortionDataset(root_dir=texture_folder, transform=data_transform)

    # Create data loaders
    batch_size = 32
    deformation_loader = DataLoader(deformation_dataset, batch_size=batch_size, shuffle=True)
    texture_loader = DataLoader(texture_dataset, batch_size=batch_size, shuffle=True)

    return [deformation_loader, texture_loader]

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
