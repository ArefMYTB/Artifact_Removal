import torch
import torchvision
import torchvision.transforms as transforms
import yaml
import os
from PIL import Image


from datasets import load_dataset
from torch.utils.data import DataLoader

from DDT.model import CoefficientGenerator
# from DDT.ddt import DDT

from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import *
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel
from ip_adapter import IPAdapter

def prepare_dataset(dataset_name):
    dataset = load_dataset(
        dataset_name,
        None,
        cache_dir=None,
    )

    def preprocess_train(examples):
        transform = transforms.ToTensor()

        prompts = examples["prompt"]

        flawless_images = [image.convert("RGB") for image in examples['flawless']]
        flawless_images = [transform(image) for image in flawless_images]

        distorted_images = [image.convert("RGB") for image in examples['distorted']]
        distorted_images = [transform(image) for image in distorted_images]

        mask_images = [image.convert("L") for image in examples['mask']]
        mask_images = [transform(image) for image in mask_images]

        segment_images = [image.convert("P") for image in examples['segment']]
        segment_images = [transform(image) for image in segment_images]

        canny_images = [image.convert("RGB") for image in examples['canny']]
        canny_images = [transform(image) for image in canny_images]

        pose_images = [image.convert("RGB") for image in examples['pose']]
        pose_images = [transform(image) for image in pose_images]       

        reference_images = [image.convert("RGB") for image in examples['reference']]
        reference_images = [transform(image) for image in reference_images] 

        examples["prompt"] = prompts
        examples["flawless"] = flawless_images
        examples["distorted"] = distorted_images
        examples["mask"] = mask_images
        examples["conditions"] = [list(t) for t in zip(canny_images, pose_images, segment_images)]
        examples["reference"] = reference_images


        return examples

    return dataset["train"].with_transform(preprocess_train)


def get_loaders(train_dataset, batch_size):
    def collate_fn(examples):
        prompt = [example["prompt"] for example in examples]
        flawless = [example["flawless"] for example in examples]
        distorted = [example["distorted"] for example in examples]
        mask = [example["mask"] for example in examples]
        conditions = [example["conditions"] for example in examples]
        reference = [example["reference"] for example in examples]

        return {
            "prompt": prompt,
            "flawless": flawless,
            "distorted": distorted,
            "mask": mask,
            "conditions": conditions,
            "reference": reference,
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

    base_model_path = conf["base_model_path"]  #"runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = conf["image_encoder_path"]  #"models/image_encoder/"
    ip_ckpt = conf["ip_ckpt"]  #"models/ip-adapter_sd15.bin"
    device = "cuda"

    # Initialize your model
    model = CoefficientGenerator(resolution=resolution, num_output=num_coefficients)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Inpaint model
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # ControlNet models
    canny_model_path = conf["canny_model_path"]
    pose_model_path = conf["pose_model_path"]
    segment_model_path = conf["segment_model_path"]

    canny_controlnet = ControlNetModel.from_pretrained(canny_model_path, torch_dtype=torch.float16)
    pose_controlnet = ControlNetModel.from_pretrained(pose_model_path, torch_dtype=torch.float16)
    segment_controlnet = ControlNetModel.from_pretrained(segment_model_path, torch_dtype=torch.float16)

    controlnet = [canny_controlnet, pose_controlnet, segment_controlnet]

    pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    pipe.to('cuda')

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    # Initialize your dataset
    train_dataset = prepare_dataset(dataset_name)

    # TODO: change to train_loader, val_loader, test_loader
    train_loader = get_loaders(train_dataset, batch_size)

    # Create the required directories
    checkpoints_path = conf["model_path"]
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    result_path = conf["result_path"]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Training loop
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            prompt = batch_data['prompt']
            distorted_images = batch_data['distorted']
            flawless_images = batch_data['flawless']
            mask_images = batch_data['mask']
            reference_images = batch_data['reference']
            conditions = batch_data['conditions']

            # get coefficients
            alpha = model(distorted_images, mask_images)

            print(alpha)

            # Convert tensors back to PIL Images
            to_pil = transforms.ToPILImage()
            dis_images = [to_pil(image) for image in distorted_images]
            msk_images = [to_pil(image) for image in mask_images]
            rfc_images = [to_pil(image) for image in reference_images]
            cnd_images = [[to_pil(image) for image in condition] for condition in conditions]

            transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # Inpainting
            result_batch = []
            for i in range(len(dis_images)):
                result = ip_model.generate(
                    pil_image=rfc_images[i], 
                    prompt=prompt[i],
                    image=dis_images[i], 
                    control_image=cnd_images[i],
                    controlnet_conditioning_scale=alpha[i][:3],
                    mask_image=msk_images[i], 
                    num_samples=1, 
                    num_inference_steps=30,
                    seed=42, 
                    strength=1.0
                )[0]

                result_batch.append(transform(result).requires_grad_(True))

            result_batch = torch.stack(result_batch)
            
            flawless_images = torch.stack(flawless_images).detach()

            # Calculate the loss
            loss = criterion(result_batch, flawless_images)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        print(f"[epoch {epoch}] Saving model...")
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f'trained_model_{epoch}.pth'))

        # print(f"[epoch {epoch}] Saving results...")
        # if not os.path.exists(os.path.join(result_path, str(epoch))):
        #     os.makedirs(os.path.join(result_path, str(epoch)))
        # for idx, res in enumerate(inpaint_result):
        #     transform = transforms.ToPILImage()
        #     img = transform(res)
        #     img.save(os.path.join(result_path, str(epoch), f"gen_{idx}.jpg"))

    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == '__main__':
    main()
