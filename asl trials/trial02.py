import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Load the Keras model
model = tf.keras.models.load_model(r"C:\Users\LR\Downloads\asl_gesture_model_final.keras")

# Alphabet mapping
alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Detection")
        self.root.geometry("800x600")

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")

        # GUI Elements
        self.video_label = Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.prediction_label = Label(root, text="Prediction: ", font=("Arial", 24))
        self.prediction_label.pack(pady=20)

        self.start_button = Button(root, text="Start", command=self.start_detection, font=("Arial", 16))
        self.start_button.pack(side=tk.LEFT, padx=20)

        self.stop_button = Button(root, text="Stop", command=self.stop_detection, font=("Arial", 16))
        self.stop_button.pack(side=tk.RIGHT, padx=20)

        self.running = False  # Flag to control the detection loop

    def preprocess_frame(self, frame):
        """Preprocess the frame for the model."""
        frame = cv2.resize(frame, (224, 224))  # Resize to model's input size
        frame = frame / 255.0  # Normalize pixel values to [0, 1]
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        return frame

    def decode_prediction(self, prediction):
        """Decode the model's output to an alphabet character."""
        class_index = prediction.argmax()
        return alphabet[class_index]

    def update_frame(self):
        """Capture and display the webcam frame with predictions."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Process frame for prediction
        processed_frame = self.preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        result_char = self.decode_prediction(prediction)

        # Display the prediction
        self.prediction_label.config(text=f"Prediction: {result_char}")

        # Convert the frame to RGB and display in GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule the next update
        self.root.after(10, self.update_frame)

    def start_detection(self):
        """Start the real-time detection."""
        self.running = True
        self.update_frame()

    def stop_detection(self):
        """Stop the real-time detection."""
        self.running = False
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()