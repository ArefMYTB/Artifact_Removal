import os
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import random_split

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


def get_dataloaders():
    # Read the necessary parameters from the config file
    with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)

    input_width = conf["data"]["input_width"]
    input_height = conf["data"]["input_height"]

    batch_size = conf["classifier"]["batch_size"]
    train_ratio = conf["classifier"]["train_ratio"]
    val_ratio = conf["classifier"]["val_ratio"]
    test_ratio = conf["classifier"]["test_ratio"]

    # Define paths to deformation and texture folders
    deformation_folder = conf["data"]["deformation_folder"]
    texture_folder = conf["data"]["texture_folder"]
    none_folder = conf["data"]["none_folder"]

    # Define transformation to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((input_width, input_height)),
        transforms.ToTensor()
    ])

    # Create dataset instances for deformation and texture
    deformation_dataset = DistortionDataset(root_dir=deformation_folder, transform=data_transform)
    texture_dataset = DistortionDataset(root_dir=texture_folder, transform=data_transform)
    none_dataset = DistortionDataset(root_dir=none_folder, transform=data_transform)

    distortion_dataset = ConcatDataset([deformation_dataset, texture_dataset, none_dataset])

    train_size = int(train_ratio * len(distortion_dataset))
    val_size = int(val_ratio * len(distortion_dataset))
    test_size = len(distortion_dataset) - train_size - val_size
    print(f"train size: {train_size}, val size: {val_size}, test size: {test_size}")
    train_dataset, val_dataset, test_dataset = \
        random_split(distortion_dataset, [train_size, val_size, test_size])

    def custom_collate_fn(batch):
        distorted = [sample['distorted'] for sample in batch]
        flawless = [sample['flawless'] for sample in batch]
        mask = [sample['mask'] for sample in batch]
        original = [sample['original'] for sample in batch]
        label = [sample['label'] for sample in batch]
        return {
            'distorted': distorted,
            'flawless': flawless,
            'mask': mask,
            'original': original,
            'label': label
        }

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # transform = transforms.ToPILImage()
    # for batch in tqdm(train_loader):
    #     print(len(batch))
    #     print(batch['label'])
    #     for sample in batch['distorted']:
    #         img = transform(sample[0])
    #         img.show()

    return train_loader, val_loader, test_loader

def main():
    get_dataloaders()

if __name__ == '__main__':
    main()
