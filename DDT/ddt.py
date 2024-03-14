from controlnet_aux.processor import Processor

# load processor from processor_id
# options are:
# ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
#  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
#  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
#  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
#  "softedge_pidinet", "softedge_pidsafe", "dwpose"]

class DDT:

    def __init__(self):
        self.canny_processor = Processor("canny")
        self.softedge_hed_processor = Processor("softedge_hed")
        self.pose_image_processor = Processor("openpose_full")
        

    def generate_conditions(self, img_batch, reference_img=None):
        control_batch = []
        for i, img in enumerate(img_batch):
            canny_image = self.canny_processor(img[0], to_pil=True)
            hed_image = self.softedge_hed_processor(img[0], to_pil=True)
            pose_image = self.pose_image_processor(img[0], to_pil=True)
            texture_image = reference_img[i]
            
            control_images = [canny_image, hed_image, pose_image] #, texture_image]

            control_batch.append(control_images)
        
        return control_batch
