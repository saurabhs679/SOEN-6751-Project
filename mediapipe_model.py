import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os

# Define the path to your model file
model_file_path = 'C:/Users/rohan/OneDrive/Desktop/HCI Project/gesture_recognizer.task'

# Use os.path.abspath to get the absolute path to the model file
absolute_model_file_path = os.path.abspath(model_file_path)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture
    try:
        # Check if the result is not None before printing
        if result is not None:
            # Ensure 'gestures' attribute is present in the result
            if hasattr(result, 'gestures') and result.gestures:
                # Loop through each gesture in the result
                for gesture in result.gestures:
                    # print('Gesture Category Name:', gesture[0].category_name)
                    latest_gesture = gesture[0].category_name

            else:
                # print("Result does not have 'gestures' or it's empty.")
                latest_gesture = ""
    except Exception as e:
        # Print any error that occurs
        print(f"An error occurred: {e}")
        latest_gesture = ""

base_options = python.BaseOptions(model_asset_path=absolute_model_file_path)
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=VisionRunningMode.LIVE_STREAM,result_callback=print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)
frame_timestamp_ms = int(round(time.time() * 1000))
recognizer = GestureRecognizer.create_from_options(options)
desired_width = 1300  # Set the desired width
desired_height = 800  # Set the desired height


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize the MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Variable to keep track of whether the tracker is initialized
tracker_initialized = False

latest_gesture = ""

def calculate_crop_size(frame_center, face_center, min_crop_size, frame):
    distance = np.linalg.norm(np.array(frame_center) - np.array(face_center))
    crop_size = int(max(min_crop_size, min(frame.shape[0], frame.shape[1]) - distance))
    crop_size += crop_size % 2  # Ensuring the crop size is even
    return crop_size, crop_size





# Initialize the webcam
cap = cv2.VideoCapture(0)

timestamp = 0

while cap.isOpened(): 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty frame")
        break
            
    timestamp += 1
    frame_skip = 5  # Process every 5th frame
    if timestamp % frame_skip == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_image, timestamp)


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(frame_rgb)

    if face_results.detections:
        bboxC = face_results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_x = x  # Align with the left edge of the bounding box
        text_y = y + h - 10  # Slightly above the bottom edge of the bounding box
        text = latest_gesture if latest_gesture else "No Gesture Detected"
        
        # Draw the text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if not tracker_initialized or not tracker.update(frame)[0]:
            tracker = cv2.TrackerMOSSE_create()  
            tracker.init(frame, (x, y, w, h))
            tracker_initialized = True

        if tracker_initialized:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                face_center = (x + w // 2, y + h // 2)
                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

                crop_width, crop_height = calculate_crop_size(frame_center, face_center, 400, frame)
                scale_factor = frame.shape[1] / crop_width

                new_x = max(face_center[0] - crop_width // 2, 0)
                new_y = max(face_center[1] - crop_height // 2, 0)
                new_x = min(new_x, frame.shape[1] - crop_width)
                new_y = min(new_y, frame.shape[0] - crop_height)

                cropped_frame = frame[new_y : new_y + crop_height, new_x : new_x + crop_width]


                resized_frame = cv2.resize(cropped_frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("MediaPipe Model with Framing", resized_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
