import os
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
import numpy as np
import shutil
import json

RESOLUTION = 512
DF_ROOT_DIR = "E:/Projects/Python Projects/Rectification/Dataset/Chibi DressCode"
DATASET_ROOT_DIR = os.path.join(DF_ROOT_DIR, "dataset")

FLAWLESS_KEYWORD    = 'flawless'
DISTORTED_KEYWORD   = 'distorted'
MASK_KEYWORD        = 'mask'
REFERENCE_KEYWORD   = 'reference'

PROMPT = "high quality, detailed, professional photograph of a woman standing"

parts = {
    5: 'skirt',
    6: 'pants',
}


def resize_with_padding(img, color):
    expected_size = max(img.width, img.height)

    colors = {
        'white': (255, 255, 255) if img.mode == "RGB" else 1,
        'black': (0, 0, 0) if img.mode == "RGB" else 0
    }

    bordered_image = Image.new(
        img.mode, 
        (expected_size, expected_size), 
        colors[color]
    )
    
    width_offset = (expected_size - img.width) // 2
    height_offset = (expected_size - img.height) // 2
    
    bordered_image.paste(img, (width_offset, height_offset))

    return bordered_image.resize((RESOLUTION, RESOLUTION))


def generate_datasets():
    img_root_dir = os.path.join(DF_ROOT_DIR, "images")
    seg_root_dir = os.path.join(DF_ROOT_DIR, "label_maps")

    img_file_list = os.listdir(img_root_dir)
    seg_file_list = os.listdir(seg_root_dir)

    counter = 0
    for seg_file in tqdm(seg_file_list):
        segm = Image.open(os.path.join(seg_root_dir, seg_file))
        segm = resize_with_padding(segm, color='black')
        segm = np.array(segm)

        img_file = seg_file.replace("4.png", "0.jpg")
        ref_file = seg_file.replace("4.png", "1.jpg")
        if (not img_file in img_file_list) or (not ref_file in img_file_list):
            continue

        img = Image.open(os.path.join(img_root_dir, img_file))
        img = resize_with_padding(img, color='white')

        ref_img = Image.open(os.path.join(img_root_dir, ref_file))
        ref_img = resize_with_padding(ref_img, color='white')

        for p in parts:
            part_exists = np.any(segm == p)
            if not part_exists:
                continue
            
            part = [[1 if element == p else 0 for element in row] for row in segm]

            width = len(part[0])
            height = len(part)

            image = Image.new("1", (width, height))

            pixels = image.load()

            for y in range(height):
                for x in range(width):
                    pixels[x, y] = part[y][x]

            dilation_img = image.filter(ImageFilter.MaxFilter)

            dilation_img.save(os.path.join(DATASET_ROOT_DIR, MASK_KEYWORD, MASK_KEYWORD, f"{counter}.jpg"))
            ref_img.save(os.path.join(DATASET_ROOT_DIR, REFERENCE_KEYWORD, REFERENCE_KEYWORD, f"{counter}.jpg"))
            img.save(os.path.join(DATASET_ROOT_DIR, FLAWLESS_KEYWORD, FLAWLESS_KEYWORD, f"{counter}.jpg"))

            counter += 1

    return counter


def main():
    # make the required directories
    if not os.path.exists(DATASET_ROOT_DIR):
        os.makedirs(DATASET_ROOT_DIR)

    if not os.path.exists(os.path.join(DATASET_ROOT_DIR, FLAWLESS_KEYWORD)):
        os.makedirs(os.path.join(DATASET_ROOT_DIR, FLAWLESS_KEYWORD, FLAWLESS_KEYWORD))

    if not os.path.exists(os.path.join(DATASET_ROOT_DIR, MASK_KEYWORD)):
        os.makedirs(os.path.join(DATASET_ROOT_DIR, MASK_KEYWORD, MASK_KEYWORD))

    if not os.path.exists(os.path.join(DATASET_ROOT_DIR, REFERENCE_KEYWORD)):
        os.makedirs(os.path.join(DATASET_ROOT_DIR, REFERENCE_KEYWORD, REFERENCE_KEYWORD))


    # Generate the dateset files
    counter = generate_datasets()


    # Zip the directories
    shutil.make_archive(os.path.join(DATASET_ROOT_DIR, FLAWLESS_KEYWORD), 'zip', os.path.join(DATASET_ROOT_DIR, FLAWLESS_KEYWORD))
    shutil.make_archive(os.path.join(DATASET_ROOT_DIR, MASK_KEYWORD), 'zip', os.path.join(DATASET_ROOT_DIR, MASK_KEYWORD))
    shutil.make_archive(os.path.join(DATASET_ROOT_DIR, REFERENCE_KEYWORD), 'zip', os.path.join(DATASET_ROOT_DIR, REFERENCE_KEYWORD))

    print("Archived all the directories!")


    # Prepare the train.jsonl file
    conf = open(os.path.join(DATASET_ROOT_DIR, "train.jsonl"),"a")

    for i in range(counter):
        line = json.dumps({
            "prompt": PROMPT,
            FLAWLESS_KEYWORD: f"{FLAWLESS_KEYWORD}/{i}.jpg",
            MASK_KEYWORD: f"{MASK_KEYWORD}/{i}.jpg",
            REFERENCE_KEYWORD: f"{REFERENCE_KEYWORD}/{i}.jpg",
        })
        conf.write(f"{line}\n")

    conf.close()

    print("Generated the train.jsonl file!")


if __name__ == '__main__':
    main()
