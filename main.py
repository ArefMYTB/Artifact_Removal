import cv2
import numpy as np

# Load the image
image_path = "data/clothes/shirts/4.jpg"
image = cv2.imread(image_path)


# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper HSV thresholds for detecting hands
lower_threshold = np.array([0, 20, 70])
upper_threshold = np.array([20, 255, 255])

# Create a binary mask based on the threshold
hand_mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

# Find contours in the mask
contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (presumed to be the hand)
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the hand region
x, y, width, height = cv2.boundingRect(largest_contour)

# Extract the hand region from the original image
hand_region = image[y:y+height, x:x+width]

# Define stretching factors
stretch_factor_x = 1.5
stretch_factor_y = 1.0

# Define the corners of the original hand region
corners = np.array([[x, y], [x + width, y], [x, y + height], [x + width, y + height]], dtype=np.float32)

# Define the corners of the stretched hand region
stretched_corners = np.array([[x, y], [x + width * stretch_factor_x, y], [x, y + height * stretch_factor_y], [x + width * stretch_factor_x, y + height * stretch_factor_y]], dtype=np.float32)

# Calculate the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(corners, stretched_corners)

# Apply the perspective transformation to the hand region
stretched_hand = cv2.warpPerspective(hand_region, matrix, (int(width * stretch_factor_x), int(height * stretch_factor_y)))

# Resize stretched_hand to match the dimensions of the target region
stretched_hand_resized = cv2.resize(stretched_hand, (width, height))

# Replace the stretched hand region in the original image
image[y:y+height, x:x+width] = stretched_hand_resized

# Display or save the result
cv2.imshow("Stretched Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = "output_stretched_image.jpg"
cv2.imwrite(output_path, image)
