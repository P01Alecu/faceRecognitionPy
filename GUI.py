"""
import tkinter as tk
from tkinter import ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # configuram stilul
        self.style = ttk.Style(self)
        self.style.theme_use("clam")  # putem folosi orice temă disponibilă

        # adaugam un label
        label = ttk.Label(self, text="Bine ai venit in aplicatia mea!")
        label.pack(padx=20, pady=20)

        # adaugam un buton
        button = ttk.Button(self, text="Apasa-ma!", command=self.button_click)
        button.pack(pady=10)

    def button_click(self):
        print("Buton apasat!")


if __name__ == '__main__':
    app = App()
    app.mainloop()
"""

from tkinter import *
import customtkinter

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #setting up theme of your app
        customtkinter.set_appearance_mode("dark")
        #setting up themes of your components
        customtkinter.set_default_color_theme("green")


        #setting window width and height
        self.geometry("300x400")

        #use ctkbutton 
        button = customtkinter.CTkButton(self, text = "Hello", command = self.button_click)
        #showing at the center of the screen
        button.place(relx = 0.5, rely = 0.5, anchor = CENTER)


    def button_click(self):
        print("The button was clicked.")



if __name__ == '__main__':
    app = App()
    app.mainloop()