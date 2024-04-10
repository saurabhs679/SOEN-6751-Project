#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from tkinter import PhotoImage

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import time
import math
import numpy as np
import subprocess
import os
import tkinter as tk
from tkinter import messagebox
import threading
from PIL import Image, ImageTk


desired_width = 1200  # Set the desired width
desired_height = 600  # Set the desired height
max_face_size = 300
project_root = os.path.dirname(os.path.abspath(__file__))
TRACK_CHANGE_TIME = 0.5
last_track_change = 0


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def video_stream(label, cap, stop_event,root):
    # Argument parsing #################################################################
    args = get_args()
    last_volume_change_time = 0  # Keep track of the last time the volume was changed
    
    VOLUME_CHANGE_INTERVAL = 0.2
    
    countdown_start_time = None

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Initialize the MOSSE tracker
    tracker = cv.legacy.TrackerMOSSE_create()

    # Variable to keep track of whether the tracker is initialized
    tracker_initialized = False

    ############# Circlular Gesture For Volume###################

    def adjust_system_volume(direction):
        # Placeholder for system volume adjustment logic
        # Implement this function based on your OS
        # direction can be "increase" or "decrease"
        global last_track_change , TRACK_CHANGE_TIME
        if current_time - last_track_change > TRACK_CHANGE_TIME:
            if direction == "increase":
                print(f"Next Track- {direction}")
                play_next_track()
                last_track_change = current_time
            elif direction == "decrease":
                print(f"Previous Track- {direction}")
                play_previous_track()
                last_track_change = current_time

        # if direction == "increase":
        #     play_next_track()
        # elif direction == "decrease":
        #     play_previous_track()
    def detect_circular_motion(point_history):
        """
        Detects if the motion in point_history is circular and its direction.
        Returns "clockwise", "anticlockwise", or None.
        """
        if len(point_history) < 10:
            return None  # Not enough points to analyze

        # Calculate the relative motion of points
        deltas = [(point_history[i][0] - point_history[i-1][0], point_history[i][1] - point_history[i-1][1]) for i in range(1, len(point_history))]
        
        # Calculate the angles of motion vectors
        angles = [math.atan2(delta[1], delta[0]) for delta in deltas]

        # Detect continuous angle change direction
        angle_diffs = [(angles[i] - angles[i-1]) % (2 * math.pi) for i in range(1, len(angles))]
        clockwise_count = sum(1 for diff in angle_diffs if 0 < diff < math.pi)
        anticlockwise_count = sum(1 for diff in angle_diffs if math.pi < diff < 2 * math.pi)

        if clockwise_count > len(angle_diffs) * 0.7:
            return "clockwise"
        elif anticlockwise_count > len(angle_diffs) * 0.7:
            return "anticlockwise"
        return None
    
    ##########################################################################

    #################Swipe Gesture For Forward And Reverse###############
    def media_control(action):
        # Placeholder for media control logic
        # Implement this function based on the desired media control, e.g., using keyboard shortcuts or API calls
        print(f"Media control action: {action}")
        # if action == "forward":
        #     play_next_track()
        # elif action == "reverse":
        #     play_previous_track()
        
    def detect_gesture(point_history):
        """
        Analyzes the pointer's movement history to detect either a circular or swipe gesture.
        Returns a tuple with the gesture type ("circular" or "swipe") and the specific gesture
        ("clockwise", "anticlockwise", "left_to_right", "right_to_left"), or (None, None) if no gesture is detected.
        """
        # Circular motion detection logic (as previously defined)
        # def detect_circular_motion():
        #     """
        #     Detects if the motion in point_history is circular and its direction.
        #     Returns "clockwise", "anticlockwise", or None.
        #     """
        #     if len(point_history) < 10:
        #         return None  # Not enough points to analyze

        #     # Calculate the relative motion of points
        #     deltas = [(point_history[i][0] - point_history[i-1][0], point_history[i][1] - point_history[i-1][1]) for i in range(1, len(point_history))]
            
        #     # Calculate the angles of motion vectors
        #     angles = [math.atan2(delta[1], delta[0]) for delta in deltas]

        #     # Detect continuous angle change direction
        #     angle_diffs = [(angles[i] - angles[i-1]) % (2 * math.pi) for i in range(1, len(angles))]
        #     clockwise_count = sum(1 for diff in angle_diffs if 0 < diff < math.pi)
        #     anticlockwise_count = sum(1 for diff in angle_diffs if math.pi < diff < 2 * math.pi)

        #     if clockwise_count > len(angle_diffs) * 0.7:
        #         return "clockwise"
        #     elif anticlockwise_count > len(angle_diffs) * 0.7:
        #         return "anticlockwise"
        #     return None
        def detect_circular_motion():
            """
            Detects if the motion in point_history is circular and its direction.
            Requires a minimum radius to reduce sensitivity to small movements.
            Returns "clockwise", "anticlockwise", or None.
            """
            if len(point_history) < 10:
                return None  # Not enough points to analyze

            # Calculate the centroid of the points
            centroid = np.mean(point_history, axis=0)

            # Calculate distances of points from the centroid
            distances = [math.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2) for p in point_history]

            # Calculate the average distance (approximate radius)
            avg_distance = sum(distances) / len(distances)

            # Define a minimum radius for the circular gesture to be considered valid
            min_radius = 20  # Adjust this value based on your requirement

            # Only proceed if the average distance meets the minimum radius requirement
            if avg_distance < min_radius:
                return None  # Gesture is too small, likely just noise or a shake

            # Calculate the relative motion of points
            deltas = [(point_history[i][0] - point_history[i-1][0], point_history[i][1] - point_history[i-1][1]) for i in range(1, len(point_history))]

            # Calculate the angles of motion vectors
            angles = [math.atan2(delta[1], delta[0]) for delta in deltas]

            # Detect continuous angle change direction
            angle_diffs = [(angles[i] - angles[i-1]) % (2 * math.pi) for i in range(1, len(angles))]
            clockwise_count = sum(1 for diff in angle_diffs if 0 < diff < math.pi)
            anticlockwise_count = sum(1 for diff in angle_diffs if math.pi < diff < 2 * math.pi)

            if clockwise_count > len(angle_diffs) * 0.7:
                return "clockwise"
            elif anticlockwise_count > len(angle_diffs) * 0.7:
                return "anticlockwise"
            return None
        

        # Swipe gesture detection logic (as previously defined)
        def detect_swipe_gesture():
           """
           Detects swipe gestures based on the pointer's movement history.
           Returns "left_to_right", "right_to_left", or None.
           """
           if len(point_history) < 5:
                return None  # Not enough points to analyze
           
           start_point = point_history[0]
           end_point = point_history[-1]

            # Calculate the distance and direction of the swipe
           dx = end_point[0] - start_point[0]
           dy = end_point[1] - start_point[1]

           swipe_distance = abs(dx)

           min_swipe_distance = 100

            # Check for a horizontal swipe with minimal vertical movement
           if swipe_distance > min_swipe_distance and abs(dx) > abs(dy) * 2:
            if dx > 0:
                return "left_to_right"
            else:
                return "right_to_left"
           return None

        # First, try detecting a circular gesture
        circular_gesture = detect_circular_motion()
        if circular_gesture:
            return ("circular", circular_gesture)

        # If no circular gesture was detected, try detecting a swipe gesture
        swipe_gesture = detect_swipe_gesture()
        if swipe_gesture:
            return ("swipe", swipe_gesture)

        # If no gesture was detected
        return (None, None)

    # def calculate_crop_size(frame_center, face_center, face_size, min_crop_size, max_face_size, frame):
    #     distance = np.linalg.norm(np.array(frame_center) - np.array(face_center))
    #     crop_size = int(max(min_crop_size, min(frame.shape[0], frame.shape[1]) - distance))
    #     crop_size += crop_size % 2  # Ensuring the crop size is even

    #     # Adjust crop size based on the face size to prevent excessive zooming when moving closer
    #     if face_size > max_face_size:
    #         scale_factor = max_face_size / face_size
    #         crop_size = int(crop_size * scale_factor)

    #     return crop_size, crop_size

    # Function to increase Volume
    def increase_volume():
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])

    # Function to decrease Volume
    def decrease_volume():
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])

    # Function to play music in Spotify
    def play_spotify():
        subprocess.run(["osascript", "-e", "tell application \"Spotify\" to play"])

    # Function to pause music in Spotify
    def pause_spotify():
        subprocess.run(["osascript", "-e", "tell application \"Spotify\" to pause"])

    # Function play next track in Spotify
    def play_next_track():
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to next track'])

    # Function to play previous track in Spotify
    def play_previous_track():
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to previous track'])

        # Function to check if music is playing in Spotify
    def is_spotify_playing():
        result = subprocess.run(["osascript", "-e", "tell application \"Spotify\" to player state as string"], capture_output=True, text=True)
        return result.stdout.strip() == "playing"

    # Function to check if music is paused in Spotify
    def is_spotify_paused():
        result = subprocess.run(["osascript", "-e", "tell application \"Spotify\" to player state as string"], capture_output=True, text=True)
        return result.stdout.strip() == "paused"
    
    # Function to close Spotify
    def close_spotify():
        subprocess.run(["osascript", "-e", "tell application \"Spotify\" to quit"])
    
    def close_application():
        root.quit()  # Stop the Tkinter event loop
        root.destroy()

    def quit_application():
        print("Quitting application...")

        # Close Spotify if needed (ensure this function is defined and works correctly)
        try:
            close_spotify()
        except Exception as e:
            print(f"Error closing Spotify: {e}")

        # Release the camera resource
        try:
            if cap.isOpened():
                cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        root.after(1000, close_application)

        # If there are any background threads, make sure they are properly stopped
        # For example, if you have a stop_event for a video streaming thread:
        # stop_event.set()

        # Safely close the Tkinter GUI window
        try:
            root.quit()  # For gracefully quitting the Tkinter mainloop
            root.destroy() 
        except Exception as e:
            print(f"Error closing GUI window: {e}")



    def calculate_crop_size(frame_center, face_center, face_size, initial_crop_size, max_face_size, frame):
        distance = np.linalg.norm(np.array(frame_center) - np.array(face_center))
        crop_size = int(max(initial_crop_size, min(frame.shape[0], frame.shape[1]) - distance))
        crop_size += crop_size % 2  # Ensuring the crop size is even

        # Introduce a smoothing factor to avoid abrupt changes in crop size
        smoothing_factor = 0.1  # Adjust this value based on desired smoothing effect
        crop_size = int((1 - smoothing_factor) * crop_size + smoothing_factor * max_face_size)

        # Adjust crop size based on the face size to prevent excessive zooming when moving closer
        if face_size > max_face_size:
            scale_factor = max_face_size / face_size
            crop_size = int(crop_size * scale_factor)

        return crop_size, crop_size

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(os.path.join(project_root, "model/keypoint_classifier/keypoint_classifier_label.csv"),
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(os.path.join(project_root, "model/keypoint_classifier/keypoint_classifier_label.csv"),
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    
                
    while not stop_event.is_set():
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        face_results = face_detection.process(debug_image)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if face_results.detections:
            bboxC = face_results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = debug_image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw a rectangle around the detected face
            cv.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not tracker_initialized or not tracker.update(debug_image)[0]:
                tracker = cv.legacy.TrackerMOSSE_create()  # Reinitialize the tracker
                tracker.init(debug_image, (x, y, w, h))
                tracker_initialized = True

        if tracker_initialized:
            success, bbox = tracker.update(debug_image)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                face_center = (x + w // 2, y + h // 2)
                frame_center = (debug_image.shape[1] // 2, debug_image.shape[0] // 2)
                face_size = max(w, h)

                crop_width, crop_height = calculate_crop_size(frame_center, face_center, face_size, 0, max_face_size, debug_image)
                scale_factor = debug_image.shape[1] / crop_width

                new_x = max(face_center[0] - crop_width // 2, 0)
                new_y = max(face_center[1] - crop_height // 2, 0)
                new_x = min(new_x, debug_image.shape[1] - crop_width)
                new_y = min(new_y, debug_image.shape[0] - crop_height)

                cropped_frame = debug_image[new_y : new_y + crop_height, new_x : new_x + crop_width]

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                        results.multi_handedness):
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        pre_processed_point_history_list = pre_process_point_history(
                            debug_image, point_history)
                        # Write to the dataset file
                        logging_csv(number, mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)

                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        # if hand_sign_id == 2:  # Point gesture
                        #     point_history.append(landmark_list[8])
                        # else:
                        #     point_history.append([0, 0])

                        if hand_sign_id == 2:  # Pointer gesture
                            point_history.append(landmark_list[8])  # Add the fingertip position of the index finger to the history
                            gesture_type, specific_gesture = detect_gesture(list(point_history))  # Detect gesture

                            if gesture_type == "circular":
                                if specific_gesture == "clockwise":
                                    adjust_system_volume("increase")
                                    countdown_start_time = None
                                elif specific_gesture == "anticlockwise":
                                    adjust_system_volume("decrease")
                                    countdown_start_time = None

                            # elif gesture_type == "swipe":
                            #     if specific_gesture == "left_to_right":
                            #         media_control("forward")
                            #     elif specific_gesture == "right_to_left":
                            #         media_control("reverse")

                        ########### Media control #############
                        current_time = time.time()
                        
                        if hand_sign_id == 4:
                            if current_time - last_volume_change_time > VOLUME_CHANGE_INTERVAL:
                                print("Increasing volume")
                                increase_volume()
                                last_volume_change_time = current_time
                                countdown_start_time = None
                        elif hand_sign_id == 7:
                            if current_time - last_volume_change_time > VOLUME_CHANGE_INTERVAL:
                                print("Decreasing volume")
                                decrease_volume()
                                last_volume_change_time = current_time
                                countdown_start_time = None
                        elif hand_sign_id == 5:
                            print("Next Track")
                            play_next_track()
                            countdown_start_time = None
                        elif hand_sign_id == 6:
                            print("Previous Track")
                            play_previous_track()
                            countdown_start_time = None
                        elif hand_sign_id == 3:
                            if countdown_start_time is None:
                                countdown_start_time = time.time()
                            else:
                                elapsed_time = time.time() - countdown_start_time
                                remaining_time = 3 - elapsed_time

                                if remaining_time > 0:
                                    text_position = (x, y + h + 20)  # Adjust 20 as needed for spacing, increase if needed

                                    if text_position[1] + 20 > debug_image.shape[0]:  # Assuming 20 pixels is the approximate height of the text
                                        text_position = (x, y - 20)

                                    cv.putText(debug_image, f"Quiting in {int(remaining_time)}", text_position,
                                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                                    print(f"Quiting in {int(remaining_time)}...")
                                else:
                                    quit_application()
                                    countdown_start_time = None
                            
                        elif hand_sign_id == 1:
                            if countdown_start_time is None:
                                # Start the countdown
                                countdown_start_time = time.time()
                            else:
                                # Countdown is already active, calculate elapsed time
                                elapsed_time = time.time() - countdown_start_time
                                remaining_time = 3 - elapsed_time
                                # if remaining_time > 0:
                                #     # Display the countdown on the image
                                #     text_position = (x, y + h + 10)  # Adjust 30 as needed for spacing
                                #     cv.putText(debug_image, f"Playing/Pausing in {int(remaining_time)}...", text_position,
                                #             cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                                #     print(f"Playing/Pausing in {int(remaining_time)}...")
                                if remaining_time > 0:
                                    # Display the countdown on the image
                                    text_position = (x, y + h + 20)  # Adjust 20 as needed for spacing, increase if needed

                                    # Check if text_position goes beyond the frame height
                                    if text_position[1] + 20 > debug_image.shape[0]:  # Assuming 20 pixels is the approximate height of the text
                                        # If text goes beyond the frame, adjust position to appear above the face box
                                        text_position = (x, y - 20)

                                    cv.putText(debug_image, f"{int(remaining_time)}...", text_position,
                                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                                    print(f"Playing/Pausing in {int(remaining_time)}")
                                else:
                                    # Countdown completed
                                    if is_spotify_playing():
                                        print("Spotify is currently playing")
                                        pause_spotify()
                                        print("Pause")
                                    elif is_spotify_paused():
                                        print("Spotify is currently paused")
                                        play_spotify()
                                        print("Play")
                                    # Reset the countdown start time
                                    countdown_start_time = None
                        else:
                            # If another gesture is detected, reset the countdown
                            countdown_start_time = None
                        #########################################

                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()

                        # Drawing part
                        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            brect,
                            handedness,
                            keypoint_classifier_labels[hand_sign_id],
                            point_history_classifier_labels[most_common_fg_id[0][0]],
                        )
                else:
                    point_history.append([0, 0])

                # debug_image = draw_point_history(debug_image,point_history)
                debug_image = draw_info(debug_image, fps, mode, number)

                try:
                    # Screen reflection #############################################################
                    resized_frame = cv.resize(cropped_frame, (desired_width, desired_height), interpolation=cv.INTER_LINEAR)
                    # cv.imshow('Hand Gesture Recognition', resized_frame)
                    debug_image = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)  # Convert to RGB
                    debug_image = Image.fromarray(debug_image)
                    debug_image = ImageTk.PhotoImage(image=debug_image)

                    # Display the debug_image on the Tkinter label
                    label.config(image=debug_image)
                    label.image = debug_image
                except Exception as e:
                    print("Error while displaying image")
                    continue
                time.sleep(0.02)
        # ret, frame = cap.read()
        # if ret:
        #     frame = cv.flip(frame, 1)  # Mirror the frame
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert to RGB
        #     frame = Image.fromarray(frame)
        #     frame = ImageTk.PhotoImage(image=frame)
        #     label.config(image=frame)
        #     label.image = frame
        # time.sleep(0.02)  # Sleep briefly to avoid high CPU usage



def main():

    args = get_args()

    cap_width = args.width
    cap_height = args.height

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    stop_event = threading.Event()
    
    def start_spotify():
        subprocess.run(["open", "-a", "Spotify"])

    def start_camera():
        global video_thread  # Make the video thread accessible globally so it can be stopped later
        stop_event.clear()  # Reset the stop_event before starting a new thread

        # Remove the Start button from the layout
        start_button.pack_forget()

        # Add the Stop button at the top, before the video frame
        stop_button.pack(side=tk.TOP, before=video_frame)
        start_spotify()
        # Start video streaming in a separate thread
        video_thread = threading.Thread(target=video_stream, args=(video_frame, cap, stop_event,root))
        video_thread.start()


    # def stop_camera():
    #     stop_event.set()  # Signal the video stream thread to stop
    #     video_thread.join()  # Wait for the video stream thread to terminate

    #     # Release the camera resource properly
    #     cap.release()

    #     # Re-initialize the camera for the next start
    #     cap_width = args.width
    #     cap_height = args.height
    #     cap = cv.VideoCapture(0)
    #     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    #     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #     # Stop button cleanup and re-display the Start button
    #     stop_button.pack_forget()  # Remove the Stop button from the layout
    #     start_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
    def close_application():
        root.quit()  # Stop the Tkinter event loop
        root.destroy()  # Close the window

    def stop_camera():
        
        if cap.isOpened():
            cap.release()
        
        subprocess.run(["osascript", "-e", "tell application \"Spotify\" to quit"])

        root.after(1000, close_application)


    def show_hints():
        # Create a top-level window for hints
        hint_window = tk.Toplevel(root)
        hint_window.title("Gesture Hints")

        # Define the desired frame size (width, height)
        frame_width, frame_height = 1000, 700  # Example frame size, adjust as needed

        # Load the hint image using PIL
        hint_image_path = os.path.join(project_root, "Beige Professional Practice Guide List For Business Employees Flyer A4.jpg")
        pil_image = Image.open(hint_image_path)

        # Calculate the aspect ratio of the image
        img_width, img_height = pil_image.size
        aspect_ratio = img_width / img_height

        # Calculate new dimensions based on the aspect ratio
        if img_width > img_height:
            new_width = frame_width
            new_height = int(frame_width / aspect_ratio)
        else:
            new_height = frame_height
            new_width = int(frame_height * aspect_ratio)

        # Resize the image to fit within the frame size while maintaining aspect ratio
        pil_image_resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert the PIL image object to a Tkinter-compatible PhotoImage object
        hint_image = ImageTk.PhotoImage(pil_image_resized)

        # Display the resized hint image
        hint_label = tk.Label(hint_window, image=hint_image)
        hint_label.image = hint_image  # Keep a reference to avoid garbage collection
        hint_label.pack(padx=10, pady=10)

        # Add a button to close the hints window
        close_button = tk.Button(hint_window, text="Close", command=hint_window.destroy)
        close_button.pack(pady=5)

    image_path = os.path.join(project_root, "smooth-stucco-wall.jpg")

    def resize_background(event):
        # Open the image file (this line can be moved to the global scope if the image doesn't change)
        image = Image.open(image_path)
        # Resize the image to the new window size
        resized_image = image.resize((event.width, event.height), Image.Resampling.LANCZOS)
        # Update the background image
        try:
            background_photo = ImageTk.PhotoImage(resized_image)
            background_label.config(image=background_photo)
            background_label.image = background_photo  
        except Exception as e:
            print(e)
            root.quit()
            root.destroy()

    # Initialize Tkinter root
    root = tk.Tk()
    root.title("Gesture Spotify Application")
    

    # Initial setup for the background image (using a placeholder size)
    background_image = Image.open(image_path)  # Update with your image path
    background_photo = ImageTk.PhotoImage(background_image.resize((800, 600), Image.Resampling.LANCZOS))  # Placeholder size and updated to use Image.Resampling.LANCZOS
    background_label = tk.Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Bind the resize event to the resize_background function
    root.bind('<Configure>', resize_background)
    # Welcome label (You can also place this at the top, above the button if preferred)
    welcome_label = tk.Label(root, text="Welcome to Gesture Spotify Control Application", font=("Helvetica", 16, "bold"), padx=10,
                         pady=10)
    welcome_label.pack()
    # Hint button on the main window, placed at the bottom right corner
    hint_button = tk.Button(root, text="Hints", command=show_hints, bg="blue")
    hint_button.place(relx=1.0, rely=1.0, anchor="se")
    # Start and Stop buttons (Define them before the video frame)
    start_button = tk.Button(root, text="Start Gesture Detection", command=start_camera, fg='green', bg='black', width=20, height=2,font=("Arial", 16, "bold"))
    stop_button = tk.Button(root, text="Stop Gesture Detection", command=stop_camera, fg='red', bg='black', width=20, height=2,font=("Arial", 16, "bold"))

    # Initially, only pack the Start button and place it at the top
    start_button.pack(side=tk.TOP)

    # Frame for video feed (Now, this comes after the button)
    video_frame = tk.Label(root)
    video_frame.pack()

    

    # Start the Tkinter mainloop
    root.mainloop()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        # csv_path = '/Users/saurabh/Documents/Courses/SOEN 6751/SOEN-6751-Project/SOEN-6751-Project/hand-gesture-recognition-mediapipe-main/model/keypoint_classifier/keypoint.csv'
        csv_path = os.path.join(project_root, "model/keypoint_classifier/keypoint.csv")
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        # csv_path = '/Users/saurabh/Documents/Courses/SOEN 6751/SOEN-6751-Project/SOEN-6751-Project/hand-gesture-recognition-mediapipe-main/model/point_history_classifier/point_history.csv'
        csv_path = os.path.join(project_root, "model/point_history_classifier/point_history.csv")
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
