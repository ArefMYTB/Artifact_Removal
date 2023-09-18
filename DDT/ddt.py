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


def ddt(img, reference_img=None):
    canny_image = controlnet_condition(img, "canny")
    hed_image = controlnet_condition(img, "softedge_hed")
    lineart_realistic = controlnet_condition(img, "lineart_realistic")
    normal_bae = controlnet_condition(img, "normal_bae") # little slow
    pose_image = controlnet_condition(img, "openpose_full")
    texture_image = reference_img

    control_images = [canny_image, pose_image, texture_image]
