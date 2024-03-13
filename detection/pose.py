
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def find_pose(img, pose):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    return results.pose_landmarks


def calculate_misalignment(landmarks1, landmarks2):
    misalignments = []
    for lm1, lm2 in zip(landmarks1.landmark, landmarks2.landmark):
        misalignments.append(np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2))
    return misalignments


# Load images
img2 = cv2.imread('image3.png')
img1 = cv2.imread('image4.jpg')

# Initialize pose detection
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Find poses
landmarks1 = find_pose(img1, pose)
landmarks2 = find_pose(img2, pose)

if landmarks1 and landmarks2:
    misalignments = calculate_misalignment(landmarks1, landmarks2)

    # Assuming you want to draw a rectangle around the entire area of misalignment
    # Find max misalignment to identify significant differences
    threshold = 0.08  # Adjust this threshold according to your needs
    significant_points = [(lm1.x, lm1.y) for lm1, lm2, mis in zip(landmarks1.landmark, landmarks2.landmark, misalignments) if mis > threshold]

    if significant_points:
        # Convert points to the original scale
        img_height, img_width, _ = img1.shape
        significant_points_scaled = [(int(x * img_width), int(y * img_height)) for x, y in significant_points]

        # Find bounding box coordinates
        x_coordinates, y_coordinates = zip(*significant_points_scaled)
        top_left = (min(x_coordinates), min(y_coordinates))
        bottom_right = (max(x_coordinates), max(y_coordinates))

        # Draw rectangle
        cv2.rectangle(img1, top_left, bottom_right, (0,255,0), 3)

        # Display the result
        cv2.imshow("Misalignment", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Could not find landmarks in one of the images.")
