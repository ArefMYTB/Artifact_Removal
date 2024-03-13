import cv2
import numpy as np

def draw_primary_difference(img1, img2, threshold=30):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection on both images
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    # Find absolute difference between the edge images
    diff = cv2.absdiff(edges1, edges2)
    _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Use morphology to dilate the diff image, making it easier to find regions
    kernel = np.ones((5,5), np.uint8)
    diff = cv2.dilate(diff, kernel, iterations = 1)

    # Find contours from the diff image to identify misaligned areas
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show results
    cv2.imshow('Original', img1)
    cv2.imshow('Diff', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Load images
img1 = cv2.imread('cloth4.jpg')
img2 = cv2.imread('cloth3.png')

draw_primary_difference(img1, img2)
