import os
import shutil
import argparse
import json

# Initialize the arg parser
parser = argparse.ArgumentParser(description='Prepare the dataset to be uploaded to the 🤗')

parser.add_argument('--root_dir', type=str, help='The directory path to the original dataset')

args = parser.parse_args()

ROOT_DIR = args.root_dir


FLAWLESS_KEYWORD    = 'flawless'
DISTORTED_KEYWORD   = 'distorted'
MASK_KEYWORD        = 'mask'
REFERENCE_KEYWORD   = 'reference'

PROMPT = "high quality, detailed, professional photograph of a woman standing"


# make the required directories
if not os.path.exists(FLAWLESS_KEYWORD):
    os.makedirs(FLAWLESS_KEYWORD)

if not os.path.exists(DISTORTED_KEYWORD):
    os.makedirs(DISTORTED_KEYWORD)

if not os.path.exists(MASK_KEYWORD):
    os.makedirs(MASK_KEYWORD)

if not os.path.exists(REFERENCE_KEYWORD):
    os.makedirs(REFERENCE_KEYWORD)


# Copy the files into their corresponding directory
counter = 0
for subdir, dirs, files in os.walk(ROOT_DIR):
    reference_images = []
    mask_image = None
    distorted_image = None
    flawless_image = None
    for file in files:
        if file.startswith(REFERENCE_KEYWORD):
            reference_images.append(file)
        elif file.startswith(MASK_KEYWORD):
            mask_image = file
        elif file.startswith(DISTORTED_KEYWORD):
            distorted_image = file
        elif file.startswith(FLAWLESS_KEYWORD):
            flawless_image = file

    if len(reference_images) == 0 or mask_image == None or \
        distorted_image == None or flawless_image == None:
        continue

    for ri in reference_images:
        shutil.copy(
            os.path.join(subdir, ri), 
            os.path.join(REFERENCE_KEYWORD, f'{counter}.jpg')
        )
        shutil.copy(
            os.path.join(subdir, mask_image), 
            os.path.join(MASK_KEYWORD, f'{counter}.jpg')
        )
        shutil.copy(
            os.path.join(subdir, distorted_image), 
            os.path.join(DISTORTED_KEYWORD, f'{counter}.jpg')
        )
        shutil.copy(
            os.path.join(subdir, flawless_image), 
            os.path.join(FLAWLESS_KEYWORD, f'{counter}.jpg')
        )
        
        counter += 1

print(f"Finished processing the dataset. Total number of samples: {counter}")


shutil.make_archive(FLAWLESS_KEYWORD, 'zip', FLAWLESS_KEYWORD)
shutil.make_archive(DISTORTED_KEYWORD, 'zip', DISTORTED_KEYWORD)
shutil.make_archive(MASK_KEYWORD, 'zip', MASK_KEYWORD)
shutil.make_archive(REFERENCE_KEYWORD, 'zip', REFERENCE_KEYWORD)

print("Archived all the directories!")


# Prepare the train.jsonl file
conf = open("train.jsonl","a")

for i in range(counter):
    line = json.dumps({
        "prompt": PROMPT,
        FLAWLESS_KEYWORD: f"{FLAWLESS_KEYWORD}/{i}.jpg",
        DISTORTED_KEYWORD: f"{DISTORTED_KEYWORD}/{i}.jpg",
        MASK_KEYWORD: f"{MASK_KEYWORD}/{i}.jpg",
        REFERENCE_KEYWORD: f"{REFERENCE_KEYWORD}/{i}.jpg",
    })
    conf.write(f"{line}\n")

conf.close()

print("Generated the train.jsonl file!")
