from controlnet_aux.processor import Processor

# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe", "dwpose"]

def controlnet_condition(img, processor_id):
    processor = Processor(processor_id)

    processed_image = processor(img, to_pil=True)
    return processed_image


def ddt(img_batch, reference_img=None):
    control_batch = []
    for i, img in enumerate(img_batch):
        canny_image = controlnet_condition(img[0], "canny")
        hed_image = controlnet_condition(img[0], "softedge_hed")
        lineart_realistic = controlnet_condition(img[0], "lineart_realistic")
        normal_bae = controlnet_condition(img[0], "normal_bae") # little slow
        pose_image = controlnet_condition(img[0], "openpose_full")
        texture_image = reference_img[i]
        
        control_images = [canny_image, hed_image, lineart_realistic, normal_bae, pose_image, texture_image]

        control_batch.append(control_images)
    
    return control_batch
