#!/usr/bin/env python
 
# Make sure you have three folders namely
# dataset, masks, mask_inv
# dataset folder must contain images with name as numbers starting from 1,2,3...
'''
Keys:
  r     - Mask the image
  SPACE - Reset the inpainting mask
  ESC   - Skip current image
  z     - Exit program
'''
from __future__ import print_function
 
import cv2  # Import the OpenCV library
import numpy as np # Import Numpy library
import sys  # Enables the passing of arguments
import os # Enables walking through the directories
import shutil
 
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)
 
    def show(self):
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.dests[0])
 
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None
 
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 25)
            self.dirty = True
            self.prev_pt = pt
            self.show()
 

def masked_region(mask, src_img):
    mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR) #change mask to a 3 channel image 
    mask_out=cv2.subtract(mask,src_img)
    mask_out=cv2.subtract(mask,mask_out)
    return mask_out


def read_file(file):
    try:
        counter_file = open(file, "r") 
        count = counter_file.read()
        counter_file.close()
        return int(count)
    except FileNotFoundError:
        count = 0
        print("Creating counter file...")
        write_file(file, count)
        return int(count)


def write_file(file, count):
    counter_file = open(file, "w")
    counter_file.write(str(count))
    counter_file.close()


# Define the path of the dataset and mask
IMG_PATH = 'DHI/'

COUNTER_FILE = "counter_file.txt"  # counter file to pause and resume mask operation
BREAK_LIMIT = 15  # Program exits after every 20 images


def main():

    # Load the image and store into a variable
    img_name = read_file(COUNTER_FILE) + 1
    OCCURRENCES = 0
    mask_count = 1
    last_data_number = 15
    FLAG = "A"
    while img_name:
        if img_name > BREAK_LIMIT:
            print('done')
            sys.exit(1)
        try:
            image = cv2.imread(IMG_PATH + str(img_name) + '/distorted' + ".png")
            # print(IMG_PATH + str(img_name))

            if image is None:
                # write_file(COUNTER_FILE, img_name-1)
                print('Failed to load image file:', image)
                sys.exit(1)

            # Create an image for sketching the mask
            image_mark = image.copy()
            sketch = Sketcher('Image', [image_mark], lambda: ((175, 160, 255), 255))

            # Sketch a mask
            while True:
                ch = cv2.waitKey()
                if ch == 27:  # ESC - next image
                    img_name += 1
                    mask_count = 1
                    break
                if ch == ord('r'):  # r - mask the image
                    FLAG = 'y'
                    break
                if ch == ord(' '):  # SPACE - reset the inpainting mask
                    image_mark[:] = image
                    sketch.show()
                if ch == ord("z"):  # z - exit program
                    # write_file(COUNTER_FILE, img_name-1)
                    print("Exited at " + str(img_name-1))
                    sys.exit(1)

            # define range of pink color in HSV
            lower_white = np.array([155, 140, 255])
            upper_white = np.array([175, 160, 255])

            # Create the mask
            border_mask = cv2.inRange(image_mark, lower_white, upper_white)
            mask_copy = border_mask.copy()

            # Perform morphology
            se = np.ones((7, 7), dtype='uint8')
            image_close = cv2.morphologyEx(mask_copy, cv2.MORPH_CLOSE, se)

            # Your code now applied to the closed image
            cnt = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            filled_mask = np.zeros(border_mask.shape[:2], np.uint8)
            cv2.drawContours(filled_mask, cnt, -1, 255, -1)

            cv2.destroyAllWindows()

            # FLAG = cv2.waitKey()
            if FLAG == "y":
                if mask_count > 1:  # create a new folder
                    # Define the folders
                    last_data_number += 1
                    destination_folder = IMG_PATH + f'{last_data_number}'
                    source_folder = IMG_PATH + str(img_name)
                    # Check if the destination folder exists, and create it if it doesn't
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)

                    for file_name in os.listdir(source_folder):
                        source_file_path = os.path.join(source_folder, file_name)
                        destination_file_path = os.path.join(destination_folder, file_name)

                        # Copy the file
                        shutil.copy2(source_file_path, destination_file_path)

                    cv2.imwrite(destination_folder + '/mask' + ".png", filled_mask)
                else:
                    cv2.imwrite(IMG_PATH + str(img_name) + '/mask' + ".png", filled_mask)

                cv2.destroyAllWindows()
                FLAG = "A"
                mask_count += 1
                continue
            else:
                cv2.destroyAllWindows()
                continue

        except Exception as e:
            print(str(e) + "\nTrying again...")
            OCCURRENCES += 1
            cv2.destroyAllWindows()
            if OCCURRENCES > 25:
                cv2.destroyAllWindows()
                break
            continue

 
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
