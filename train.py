import torch
import torchvision.transforms as transforms
import yaml

from datasets import load_dataset
from torch.utils.data import DataLoader

from DDT.model import CoefficientGenerator
# from DDT.ddt import DDT
from ControlNet.pipeline import InpaintPipeline

# get coefficient model from model.py in DDT ✅

# get dataset from data_prepare.py in Data ✅

# training loop ✅

# run the model and generate coefficients ✅

# get conditions from ddt.py in DDT ✅

# inpaint with pipeline.py in ControlNet ✅

# Loss on problem source

# Loss on coefficient generator ✅


def prepare_dataset(dataset_name):
    dataset = load_dataset(
        dataset_name,
        None,
        cache_dir=None,
    )

    def preprocess_train(examples):
        transform = transforms.ToTensor()

        prompts = examples["prompt"]

        distorted_images = [image.convert("RGB") for image in examples['distorted']]
        distorted_images = [transform(image) for image in distorted_images]

        flawless_images = [image.convert("RGB") for image in examples['flawless']]
        flawless_images = [transform(image) for image in flawless_images]

        mask_images = [image.convert("L") for image in examples['mask']]
        mask_images = [transform(image) for image in mask_images]

        # reference_images = [image.convert("RGB") for image in examples['reference']]
        # reference_images = [transform(image) for image in reference_images]

        canny_images = [image.convert("RGB") for image in examples['canny']]
        canny_images = [transform(image) for image in canny_images]

        hed_images = [image.convert("RGB") for image in examples['hed']]
        hed_images = [transform(image) for image in hed_images]

        pose_images = [image.convert("RGB") for image in examples['pose']]
        pose_images = [transform(image) for image in pose_images]        

        examples["distorted"] = distorted_images
        examples["flawless"] = flawless_images
        examples["mask"] = mask_images
        # examples["reference"] = reference_images
        examples["conditions"] = [list(t) for t in zip(canny_images, hed_images, pose_images)]
        examples["prompt"] = [list(t) for t in zip(prompts, prompts, prompts)]

        return examples

    return dataset["train"].with_transform(preprocess_train)


def get_loaders(train_dataset, batch_size):
    def collate_fn(examples):
        prompt = [example["prompt"] for example in examples]
        distorted = [example["distorted"] for example in examples]
        flawless = [example["flawless"] for example in examples]
        mask = [example["mask"] for example in examples]
        # reference = [example["reference"] for example in examples]
        conditions = [example["conditions"] for example in examples]

        return {
            "prompt": prompt,
            "distorted": distorted,
            "flawless": flawless,
            "mask": mask,
            # "reference": reference,
            "conditions": conditions,
        }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    return train_dataloader


def main():
    # Read the necessary parameters from the config file
    with open("config.yml", 'r') as file:
        conf = yaml.safe_load(file)["model"]
        
    learning_rate = conf["learning_rate"]
    num_epochs = conf["num_epochs"]
    num_coefficients = conf["num_coefficients"]
    batch_size = conf["batch_size"]

    dataset_name = conf["dataset_name"]
    resolution = conf["resolution"]

    # Initialize your model
    model = CoefficientGenerator(resolution=resolution, num_output=num_coefficients)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Initialize your dataset
    train_dataset = prepare_dataset(dataset_name)
    
    # TODO: change to train_loader, val_loader, test_loader
    train_loader = get_loaders(train_dataset, batch_size)

    inpaint_pipeline = InpaintPipeline()
    # ddt = DDT()

    # Training loop
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            prompt = batch_data['prompt']
            distorted_images = batch_data['distorted']
            flawless_images = batch_data['flawless']
            mask_images = batch_data['mask']
            # reference_images = batch_data['reference']
            conditions = batch_data['conditions']
          
            # get coefficients
            alpha = model(distorted_images, mask_images)

            # get conditions
            # conditions = ddt.generate_conditions(distorted_images, reference_images)

            # Inpainting
            inpaint_result = inpaint_pipeline.inpaint(distorted_images, mask_images, conditions, prompt, alpha)

            flawless_images = torch.stack(flawless_images).detach()

            # Calculate the loss
            loss = criterion(inpaint_result, flawless_images)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

            # TODO set save_result and result_path in config
            save_result = 2
            result_path = conf["result_path"]
            if epoch == save_result:
                print(f"[epoch {epoch}] Saving results...")
                inpaint_result.save(os.path.join(result_path, f"_{epoch}"))

            # TODO set save_model and checkpoints_path in config
            save_model = 2
            checkpoints_path = conf["model_path"]
            if epoch == save_model:
                print("[epoch {epoch}] Saving model...")
                torch.save(model.state_dict(), os.path.join(checkpoints_path, f'trained_model_{epoch}.pth'))

    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    main()
