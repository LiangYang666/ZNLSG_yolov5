import tkinter as tk
from PIL import Image, ImageTk

top = tk.Tk()

top.title("watch images")
# width = 1000
# height = 1000
# top.geometry(f'{width}x{height}')

img_open = Image.open("../data/images/bus.jpg")
img_gif = ImageTk.PhotoImage(img_open)
label_img = tk.Label(top, image=img_gif)
label_img.place(x=30, y=60)




def change_img():
    global img_gif
    img_open = Image.open("../data/images/zidane.jpg")
    img_gif = ImageTk.PhotoImage(img_open)
    label_img.configure(image=img_gif)
    # label_img.place(x=30, y=120)

button = tk.Button(top, text='Next', command=change_img)
button.place(x=60, y=20)

top.mainloop()