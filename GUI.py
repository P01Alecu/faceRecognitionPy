from tkinter import *
import tkinter as tk
import customtkinter
from tkinter_webcam import webcam
import cv2 
import os
import PIL

from testData import ModelTester

class CustomWebcamBox(tk.Frame):
    def __init__(self, master=None, tester=None, width=None, height=None, bg=None, **kwargs):
        super().__init__(master, width=width, height=height, bg=bg, **kwargs)
        self.tester = tester
        self.video = cv2.VideoCapture(0)
        self.update_frames_with_predictions()

    def update_frames_with_predictions(self):
        ret, frame = self.video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(frame)
        image = image.resize((self.winfo_width(), self.winfo_height()), PIL.Image.ANTIALIAS)
        image_tk = PIL.ImageTk.PhotoImage(image=image)
        self.picture = image_tk
        self.image_tk = image_tk
        self._update_loop()

    def _update_loop(self):
        ret, frame = self.video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(frame)
        image = image.resize((self.winfo_width(), self.winfo_height()), PIL.Image.ANTIALIAS)
        image_tk = PIL.ImageTk.PhotoImage(image=image)
        self.picture = image_tk
        self.image_tk = image_tk
        self.show_method(frame)
        self.after(10, self._update_loop)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Facial emotion detection")
        
        #setting window width and height
        self.geometry("1280x720")
        self.minsize(720, 480)
        #setting up theme of your app
        customtkinter.set_appearance_mode("dark")
        #setting up themes of your components
        customtkinter.set_default_color_theme("green")

        # Create a frame for the buttons
        button_frame = Frame(self, bg="black")
        button_frame.place(relx=0.8, rely=0.2)

        # Create the buttons
        self.button = customtkinter.CTkButton(button_frame, text="Start emotion recognition", command=self.recognitionClick)
        self.button.pack(pady=10)
        
        self.btn2 = customtkinter.CTkButton(button_frame, text="Btn 2", command=self.button_click)
        self.btn2.pack(pady=10)

        self.btn3 = customtkinter.CTkButton(button_frame, text="Btn 3", command=self.button_click)
        self.btn3.pack(pady=10)

        self.btn4 = customtkinter.CTkButton(button_frame, text="Btn 4", command=self.button_click)
        self.btn4.pack(pady=10)

        # Create a frame for the video
        video_frame = Frame(self, bg="black")
        video_frame.place(relx=0.05, rely=0.1)

        # Uses Box class from webcam to create video window

        self.tester = ModelTester('model_file_100epochs.h5', 'data/test', (48,48), 7)
        video = CustomWebcamBox(video_frame, self.tester, width=720, height=576)
        video.place(x=0, y=0)

        #video = webcam.Box(video_frame, width=720, height=576)
        #video.show_frames()  # Show the created Box

    def button_click(self):
        print("The button was clicked.")
    
    def recognitionClick(self):
        tester = ModelTester('model_file_100epochs.h5', 'data/test', (48,48), 7)
        tester.predict_image('happyFamily.jpg')



if __name__ == '__main__':
    app = App()
    app.mainloop()
