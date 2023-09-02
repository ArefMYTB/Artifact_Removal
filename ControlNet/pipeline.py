from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import *
# from diffusers import DDIMScheduler
# from diffusers.utils import load_image


# load controlnet and stable diffusion v1-5-inpainting
def load_controlnet():
    # load conditions: HED, Pose, Texture, Reference
    controlnet1 = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16
    )
    controlnet2 = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
    )
    # TODO load other condition pretraiend model
    # controlnet3 = []
    # controlnet4 = []
    controlnet = [controlnet1, controlnet2, controlnet1, controlnet2]

    # controlnet pipeline
    controlNetInpaintPipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
    )

    # controlNetInpaintPipeline.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # controlNetInpaintPipeline.enable_xformers_memory_efficient_attention()
    return controlNetInpaintPipeline


def inpaint(distorted_image, mask_image, conditions, alpha):
    # get text-prompt and num_inference_steps from config
    num_inference_steps = 30
    text_prompt = ""

    # load controlnet pretrained conditions and Stable-Diffusion-Inpainting
    controlNetInpaintPipeline = load_controlnet()
    controlNetInpaintPipeline.to('cuda')

    # generate image
    generator = torch.manual_seed(0)

    result = controlNetInpaintPipeline(
        text_prompt=text_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        image=distorted_image,
        mask_image=mask_image,
        control_image=conditions,
        controlnet_conditioning_scale=alpha
    ).images[0]

    return result

