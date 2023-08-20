import cv2
import numpy as np

# Initialize variables
drawing = False
mode = True
ix, iy = -1, -1

image_path = 'input.png'
output_path = 'mask.png'

# Mouse callback function
def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), 10, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mask, (x, y), 10, (0, 0, 0), -1)

def mask_generator(mask_path, img):

    # Create a mask for black areas
    black_mask = (img == 0)

    # Create a white image of the same size
    white_img = np.ones_like(img) * 0

    # Set black areas to white in the white image
    white_img[black_mask] = 255

    cv2.imwrite(mask_path, white_img)  # Save the output image

# Load an image
image = cv2.imread(image_path)
mask = image.copy()

# Create a window and set the mouse callback function
cv2.namedWindow('Image Masking')
cv2.setMouseCallback('Image Masking', draw_mask)

while True:
    cv2.imshow('Image Masking', mask)
    key = cv2.waitKey(1) & 0xFF

    # Clear the mask when 'c' is pressed
    if key == ord('c'):
        mask = image.copy()

    # Exit the loop when 'q' is pressed
    elif key == ord('q'):
        break

# save mask on the original image
# cv2.imwrite(output_path, mask)
# save only the mask
mask_generator(output_path, mask)

cv2.destroyAllWindows()

