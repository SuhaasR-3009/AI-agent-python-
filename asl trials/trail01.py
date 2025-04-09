import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pyttsx3
import tensorflow as tf
import numpy as np
import cv2

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("500x400")
        self.root.configure(bg="#f5f5f5")

        # Initialize Text-to-Speech Engine
        self.engine = pyttsx3.init()

        # Load the Keras model
        self.model = tf.keras.models.load_model(r"C:\Users\LR\Downloads\asl_gesture_model_final.keras")

        # Title label
        self.title_label = tk.Label(root, text="Sign Language Translator", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#333")
        self.title_label.pack(pady=10)

        # Text display area
        self.text_area = tk.Text(root, height=10, width=50, font=("Arial", 12))
        self.text_area.pack(pady=10)

        # Button frame
        self.button_frame = tk.Frame(root, bg="#f5f5f5")
        self.button_frame.pack(pady=10)

        # Buttons
        self.start_button = ttk.Button(self.button_frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, padx=10)

        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=1, padx=10)

        self.speak_button = ttk.Button(self.button_frame, text="Speak", command=self.speak)
        self.speak_button.grid(row=0, column=2, padx=10)

    def start(self):
        try:
            # Capture image from webcam
            image_data = self.get_image_input()

            # Make prediction using the Keras model
            prediction = self.model.predict(image_data)

            # Convert prediction to meaningful text
            result_text = self.decode_prediction(prediction)

            # Display result in the text area
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def reset(self):
        # Reset the text area
        self.text_area.delete(1.0, tk.END)
        messagebox.showinfo("Reset", "Text area cleared.")

    def speak(self):
        # Convert text to speech
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            messagebox.showwarning("Speak", "Text area is empty. Please add text.")

    def get_image_input(self):
        # Open webcam and capture a single frame
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam.")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Failed to capture image from webcam.")

        # Preprocess the frame for the model
        frame = cv2.resize(frame, (224, 224))  # Resize to model's input size
        frame = frame / 255.0  # Normalize pixel values to [0, 1]
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        return frame

    def decode_prediction(self, prediction):
        # Example: Decode prediction to human-readable text
        # Replace this with your own decoding logic
        classes = ["Hello", "Thanks", "Yes", "No"]  # Example class labels
        class_index = prediction.argmax()
        return classes[class_index]

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
