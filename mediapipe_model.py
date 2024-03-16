import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_gesture = ""

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture
    try:
        # Check if the result is not None before printing
        if result is not None:
            # Ensure 'gestures' attribute is present in the result
            if hasattr(result, 'gestures') and result.gestures:
                # Loop through each gesture in the result
                for gesture in result.gestures:
                    print('Gesture Category Name:', gesture[0].category_name)
                    latest_gesture = gesture[0].category_name

            else:
                print("Result does not have 'gestures' or it's empty.")
                latest_gesture = ""
    except Exception as e:
        # Print any error that occurs
        print(f"An error occurred: {e}")
        latest_gesture = ""


base_options = python.BaseOptions(model_asset_path='C:/Users/rohan/OneDrive/Desktop/HCI Project/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options,running_mode=VisionRunningMode.LIVE_STREAM,result_callback=print_result)
recognizer = vision.GestureRecognizer.create_from_options(options)
frame_timestamp_ms = int(round(time.time() * 1000))
# Initialize the webcam
cap = cv2.VideoCapture(0)

timestamp = 0

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened(): 
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        recognizer.recognize_async(mp_image, timestamp)

        if latest_gesture:
            cv2.putText(frame, latest_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("MediaPipe Model", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
