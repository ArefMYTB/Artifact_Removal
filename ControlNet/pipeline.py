import torch
import torchvision.transforms as transforms
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import *
# from diffusers import DDIMScheduler
# from diffusers.utils import load_image

class InpaintPipeline:

    # load controlnet and stable diffusion v1-5-inpainting
    def __init__(self):
        # get text-prompt and num_inference_steps from config
        self.num_inference_steps = 25

        # load conditions: HED, Pose, Texture, Reference
        self.controlnet3 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        self.controlnet1 = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
        self.controlnet2 = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
        # self.controlnet4 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16)
        # self.controlnet5 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16)

        self.controlnet = [self.controlnet1, self.controlnet2, self.controlnet3] #, self.controlnet4, self.controlnet5]
        
        self.controlNetInpaintPipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE", controlnet=self.controlnet, torch_dtype=torch.float16
        )

        self.controlNetInpaintPipeline.to('cuda')

        # controlNetInpaintPipeline.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # controlNetInpaintPipeline.enable_xformers_memory_efficient_attention()


    def inpaint(self, distorted_image, mask_image, conditions, prompt, alpha):        
        # generate image
        generator = torch.manual_seed(0)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        result_batch = []
        for i in range(len(distorted_image)):
            result = self.controlNetInpaintPipeline(
                prompt[i],
                num_inference_steps=self.num_inference_steps,
                generator=generator,
                image=distorted_image[i],
                mask_image=mask_image[i],
                control_image=conditions[i],
                controlnet_conditioning_scale=alpha[i]
            ).images[0]

            result_batch.append(transform(result).requires_grad_(True))

        result_batch = torch.stack(result_batch)
        return result_batch

