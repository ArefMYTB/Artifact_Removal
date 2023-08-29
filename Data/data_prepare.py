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

        distorted_image = None
        flawless_image = None
        mask_image = None
        original_images = []
        
        for file in os.listdir(folder_path):
            if file.startswith("distorted"):
                distorted_image = Image.open(os.path.join(folder_path, file))
            if file.startswith("flawless"):
                flawless_image = Image.open(os.path.join(folder_path, file))
            elif file.startswith("mask"):
                mask_image = Image.open(os.path.join(folder_path, file))
                mask_image = mask_image.convert('L')
            elif file.startswith("original"):
                original_img = Image.open(os.path.join(folder_path, file))
                original_images.append(original_img)

        if self.transform:
            distorted_image = self.transform(distorted_image)
            flawless_image = self.transform(flawless_image)
            mask_image = self.transform(mask_image)
            original_images = [self.transform(img) for img in original_images]

        sample = {
            'distorted': distorted_image,
            'flawless': flawless_image,
            'mask': mask_image,
            'original': original_images,
            'label': self.root_dir.split('/')[-1]
        }

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
    # for batch in tqdm(test_loader):
    #     print(len(batch))
    #     print(batch['label'])
    #     for sample in batch['mask']:
    #         img = transform(sample)
    #         img.show()

    return train_loader, val_loader, test_loader

def main():
    get_dataloaders()

if __name__ == '__main__':
    main()
