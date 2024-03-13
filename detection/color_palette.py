
import cv2


def draw_primary_color_difference(img1_path, img2_path, threshold_value=30):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to LAB color space
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Compute absolute difference
    diff = cv2.absdiff(lab1, lab2)

    # Convert difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Thresholding to get binary mask of differences
    _, thresh = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Draw the rectangle on the original image
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original image with the rectangle drawn
    cv2.imshow("Primary Color Difference", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img1_path = 'distorted.jpg'
img2_path = 'flawless.png'

draw_primary_color_difference(img1_path, img2_path)

