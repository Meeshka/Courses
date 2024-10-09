import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic and OpenCV
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Set up holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe processes RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)

    # Draw the full-body skeleton
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Display the frame with the skeleton overlay
    cv2.imshow('Body Skeleton Tracking', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
