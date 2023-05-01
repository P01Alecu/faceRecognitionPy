from tkinter import *
import customtkinter
from tkinter_webcam import webcam
import cv2 
import os


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


        ## create 2x2 grid system
        #self.grid_rowconfigure(0, weight=1)
        #self.grid_columnconfigure(1, weight=3)
#
        #self.button = customtkinter.CTkButton(self, text = "Click me", command = self.button_click)
        #self.button.grid(row=0, column=1, padx = 2, pady=20, sticky="e")
#
        #use ctkbutton 
        self.button = customtkinter.CTkButton(self, text = "Click me", command = self.button_click)
        #showing at the center of the screen
        self.button.pack(padx=20, pady=20, side = "right")
        #button.grid(column=0, row=0, sticky=tk.N+tk.S+tk.W+tk.E)

        self.btn2 = customtkinter.CTkButton(self, text = "Btn 2", command = self.button_click)
        #showing at the center of the screen
        self.btn2.pack(padx=20, pady=20, side = "right")

        # Uses Box class from webcam to create video window
        video = webcam.Box(self, width=720, height=576)
        video.show_frames()  # Show the created Box






    def button_click(self):
        print("The button was clicked.")




if __name__ == '__main__':
    app = App()
    app.mainloop()