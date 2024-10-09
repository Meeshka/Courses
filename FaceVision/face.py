import cv2
import mediapipe as mp

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load the image
image = cv2.imread('lying_body.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose detection
results = pose.process(image_rgb)

# Define anatomical skeleton connections (bones)
skeleton_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # Collarbone
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),  # Hip
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)
]

# Draw the skeleton on the image
if results.pose_landmarks:
    for connection in skeleton_connections:
        start = results.pose_landmarks.landmark[connection[0]]
        end = results.pose_landmarks.landmark[connection[1]]

        # Convert normalized coordinates to pixel values
        h, w, _ = image.shape
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))

        # Draw the line for bones
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Red color for bones

    # Draw joints as circles
    for landmark in results.pose_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green color for joints

# Resize the image to fit the screen size
screen_width = 1024  # Example screen width
screen_height = 683  # Example screen height
resized_image = cv2.resize(image, (screen_width, screen_height))

# Display the result
cv2.imshow('Medical Skeleton Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the resized image
cv2.imwrite('skeleton_image_medical_resized.jpg', resized_image)
