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
        reference_images = []

        for file in os.listdir(folder_path):
            if file.startswith("distorted"):
                distorted_image = Image.open(os.path.join(folder_path, file))
            if file.startswith("flawless"):
                flawless_image = Image.open(os.path.join(folder_path, file))
            elif file.startswith("mask"):
                mask_image = Image.open(os.path.join(folder_path, file))
                mask_image = mask_image.convert('L')
            elif file.startswith("reference"):
                reference_img = Image.open(os.path.join(folder_path, file))
                reference_images.append(reference_img)

        if self.transform:
            distorted_image = self.transform(distorted_image)
            flawless_image = self.transform(flawless_image)
            mask_image = self.transform(mask_image)
            reference_images = [self.transform(img) for img in reference_images]

        sample = {
            'distorted': distorted_image,
            'flawless': flawless_image,
            'mask': mask_image,
            'reference': reference_images,
            'label': self.root_dir.split('/')[-1]
        }

        return sample


def get_dataloaders():
    # Read the necessary parameters from the config file
    with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)

    input_width = conf["data"]["input_width"]
    input_height = conf["data"]["input_height"]

    batch_size = conf["model"]["batch_size"]
    train_ratio = conf["model"]["train_ratio"]
    val_ratio = conf["model"]["val_ratio"]
    test_ratio = conf["model"]["test_ratio"]

    # Define paths to deformation and texture folders
    dataset_folder = conf["data"]["DHI_folder"]

    # Define transformation to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((input_width, input_height)),
        transforms.ToTensor()
    ])

    distortion_dataset = DistortionDataset(root_dir=dataset_folder, transform=data_transform)

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
        reference = [sample['reference'] for sample in batch]
        label = [sample['label'] for sample in batch]
        return {
            'distorted': distorted,
            'flawless': flawless,
            'mask': mask,
            'reference': reference,
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

    print("Dataset loaded")
    return train_loader, val_loader, test_loader


def main():
    get_dataloaders()


if __name__ == '__main__':
    main()
