import tkinter as tk
import cv2
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SOEN 6751: HUMAN COMPUTER INTERFACE DESGIN (Advanced Gesture Control)")
        self.root.geometry("1920x1080")  # Set window size to 800x600

        self.start_button = tk.Button(root, text="Start", command=self.open_camera)
        self.start_button.pack(pady=20)

    def open_camera(self):
        cap = cv2.VideoCapture(0)

        self.panel = tk.Label(self.root)
        self.panel.pack()

        def show_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1000, 600))  # Resize frame to fit window
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.panel.configure(image=frame)
                self.panel.image = frame
                self.panel.after(10, show_frame)
            else:
                cap.release()  # Release the camera capture object when done

        show_frame()

root = tk.Tk()
app = CameraApp(root)
root.mainloop()
