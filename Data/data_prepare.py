import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml

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
            'original': original_images,
            'label': self.root_dir.split('/')[-1]
        }

        if self.transform:
            sample['distorted'] = [self.transform(img) for img in sample['distorted']]
            sample['flawless'] = [self.transform(img) for img in sample['flawless']]
            sample['mask'] = [self.transform(img) for img in sample['mask']]
            sample['original'] = [self.transform(img) for img in sample['original']]

        return sample


def get_dataloader():
    # Read the necessary parameters from the config file
    with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)["data"]

    input_width = conf["input_width"]
    input_height = conf["input_height"]

    batch_size = conf["batch_size"]

    # Define paths to deformation and texture folders
    deformation_folder = conf["deformation_folder"]
    texture_folder = conf["texture_folder"]

    # Define transformation to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((input_width, input_height))
    ])

    # Create dataset instances for deformation and texture
    deformation_dataset = DistortionDataset(root_dir=deformation_folder, transform=data_transform)
    texture_dataset = DistortionDataset(root_dir=texture_folder, transform=data_transform)
    distortion_dataset = ConcatDataset([deformation_dataset, texture_dataset])

    # Create data loaders
    distortion_loader = DataLoader(distortion_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # for batch in tqdm(distortion_loader):
    #     print(len(batch))
    #     for sample in batch:
    #         img = sample['distorted'][0]
    #         img.show()

    return distortion_loader

def main():
    get_dataloader()

if __name__ == '__main__':
    main()
