import customtkinter as customTk
from PIL import Image
import tkinter.filedialog as tkFileDialog
import matplotlib


# sounds
from playsound import playsound


# statistics
matplotlib.use("TkAgg")  # Set the backend for compatibility
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# import adaline file
from src.utilities import *
from src.adaline import *


# app and screen configurations

# Import fonts
customTk.FontManager.load_font("./fonts/Ara Hamah Homs.ttf")
customTk.FontManager.load_font("./fonts/NotoSansMath-Regular.ttf")
# create window using customtkinter
app = customTk.CTk()
app.bg = "red"
# set full screen size
window_size = (900, 600)
# disable resizing
app.resizable(width=False, height=False)
# set height and width
app.geometry(f"{window_size[0]}x{window_size[1]}")
# window name
app.title("Cat or Dog")
# mode dark, light or system default
customTk.set_appearance_mode("light")
# app icon
app.iconbitmap("./assests/icons/icon.ico")
# app background image
background_image = customTk.CTkImage(
    light_image=Image.open("./assests/images/GBackground.png"), size=window_size
)
# app header
header_image = customTk.CTkImage(
    light_image=Image.open("./assests/images/ADALINE.png"), size=(300, 50)
)
# Import fonts
customTk.FontManager.load_font("./fonts/Ara Hamah Homs.ttf")
# colors
buttons_color = "#01876E"
# font name
font_name = "Ara Hamah Homs"
# font size
font_Size = 30
# default image
imgSz = (275, 200)
# buttons y position
bt_y = 370
# check image is inserted
img_path = ""
# check train is done
train_done = False


window_frame = customTk.CTkFrame(
    master=app,
    width=window_size[0],
    height=window_size[1],
    fg_color="#93ECE0",
    corner_radius=0,
)
window_frame.place(relwidth=1, relheight=1)

header_label = customTk.CTkLabel(app, text="", image=header_image, bg_color="#93ECE0")
header_label.place(y=0, x=28)


# create tabs
tab_size = (window_size[0] - 10, window_size[1] - 20)
tab_view = customTk.CTkTabview(
    master=window_frame,
    width=tab_size[0],
    height=tab_size[1],
    fg_color="transparent",
    border_width=0,
    corner_radius=10,
    segmented_button_selected_color=buttons_color,
    segmented_button_selected_hover_color="#02C19D",
)
tab_view.pack(
    pady=0,
)

main_tab = tab_view.add("Main")  # add tab at the end
statistics_tab = tab_view.add("Learning Rate")  # add tab at the end


tab_view.set("Main")  # set currently visible tab


# set background image for each tab
bg_frame = customTk.CTkLabel(
    master=main_tab, text="", image=background_image, corner_radius=10
).place(relwidth=1, relheight=1)
bg_frame = customTk.CTkLabel(
    master=statistics_tab,
    text="",
    # image=background_image,
    corner_radius=10,
    bg_color="#93ECE0",
).place(relwidth=1, relheight=1)


# main tab

# image frame
defaultImg = customTk.CTkImage(
    light_image=Image.open("./assests/images/Default.png"),
    size=(200, 200),
)

label = customTk.CTkLabel(
    master=main_tab,
    text="",
    fg_color="white",
    bg_color="#93ece0",
    corner_radius=20,
    width=320,
    height=240,
    image=defaultImg,
)
label.place(relx=(0.5 - 0.18), y=60)

# result label
res_label = customTk.CTkLabel(
    master=main_tab,
    text="",
    fg_color="white",
    bg_color="white",
    text_color=buttons_color,
    corner_radius=20,
    font=(font_name, 50),
    width=320,
    height=70,
    anchor="center",
)

res_label.place(y=bt_y - 90, x=tab_size[0] / 2 - 168)


def writeText(label, text, font_sz=50, color=buttons_color):
    label.configure(text=text, text_color=color, font=(font_name, font_sz))


def catVSdog(isCat):
    if isCat:
        writeText(res_label, "Cat")
        playsound("./sounds/Cat-Sound.wav")
    else:
        writeText(res_label, "Dog")
        playsound("./sounds/Dog-Sound.wav")


# set image to label or screen
def setImage():
    global image, img_path
    res_label.configure(text="")
    path = tkFileDialog.askopenfilename(filetypes=[("Image Files", ".jpg .png .gif")])
    # Check if a file was selected
    if path=="":
        print("No file selected. Resetting to default image.")
        label.configure(image=defaultImg)  # Reset to default image
        img_path = "" # Reset img_path
        return None  # Exit the function if no file is selected

    my_image = customTk.CTkImage(
        light_image=Image.open(path),
        size=imgSz,
    )
    label.configure(image=my_image)
    img_path = path
    return path


# buttons
def center_btn(btn):
    btn_width = btn.cget("width")
    h_width = btn_width / 2
    res = h_width / tab_size[0]
    return res


# set image button function
def setImageButton():
    global img_path
    img_path = setImage()
    if img_path:
        print("image set is done")


# create the button
setImageBtn = customTk.CTkButton(
    master=main_tab,
    text="Upload Your Image",
    font=(font_name, font_Size),
    height=30,
    width=320,
    fg_color=buttons_color,
    text_color="white",
    bg_color="#93ece0",
    command=setImageButton,
    # hover=False,
    hover_color="#016653",
    corner_radius=50,
)
setImageBtn.place(relx=(0.5 - center_btn(setImageBtn)), y=bt_y)


# train button function
def trainingButton():
    global train_done
    if not train_done:
        train_done = training()
        sto_draw(statistics_frame, cost)
        writeText(res_label, "Training done", 30, "red")
    else:
        writeText(res_label, "Training has already done!", 30, "red")


# create the button
trainBtn = customTk.CTkButton(
    master=main_tab,
    text="Train",
    font=(font_name, font_Size),
    height=30,
    width=150,
    fg_color=buttons_color,
    text_color="white",
    bg_color="#93ece0",
    command=trainingButton,
    # hover=False,
    hover_color="#016653",
    corner_radius=50,
)
trainBtn.place(relx=(0.5 - center_btn(setImageBtn)), y=bt_y + 60)


def identifyingButton():
    if img_path == "" or img_path is None:
        writeText(res_label, "Please upload an image first", 30, "red")
        return

    if not train_done:
        writeText(res_label, "Please train the model first", 30, "red")
        return
   
    # Run the neural network and display the result
    result = neural(img_path)
    if result is not None:
        catVSdog(result)
    else:
        writeText(res_label, "Error identifying the image", 30, "red")


# create the button
identifyBtn = customTk.CTkButton(
    master=main_tab,
    text="Identify",
    font=(font_name, font_Size),
    height=30,
    width=150,
    fg_color=buttons_color,
    bg_color="#93ece0",
    text_color="white",
    command=identifyingButton,
    # hover=False,
    hover_color="#016653",
    corner_radius=50,
)
identifyBtn.place(relx=(0.5 + 0.018), y=bt_y + 60)


# statistics tab


statistics_frame = customTk.CTkFrame(
    master=statistics_tab,
    width=(window_size[0] - 250),
    height=(window_size[1] - 100),
)
statistics_frame.place(
    x=0,
    y=30,
)

# alpha in screen
alpha_label = customTk.CTkLabel(
    master=statistics_tab,
    width=20,
    height=20,
    bg_color="transparent",
    text=f"Alpha is : {alfa} ",
    # text_color=buttons_color,
    text_color="black",
    font=("Noto Sans Math", 30),
)
alpha_label.place(relx=0.77, y=70)


# resize button
def resizeBtn():
    global train_done
    resize(cat_dir, dog_dir, resize_path)
    train_done = False
    resizeBtn.configure(text="Resized done !")


resizeBtn = customTk.CTkButton(
    master=statistics_tab,
    text="Resize images",
    font=(font_name, font_Size),
    height=30,
    width=50,
    fg_color=buttons_color,
    text_color="white",
    bg_color="#93ece0",
    command=resizeBtn,
    # hover=False,
    hover_color="#016653",
    # corner_radius=50,
)
resizeBtn.place(relx=0.8, y=470)


def sto_draw(root, cost):
    fig = draw(cost, (statistics_size[0], statistics_size[1]))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="x", expand=False)
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    # remove unwanted items
    toolbar.children["!button4"].pack_forget()
    toolbar.update()
    toolbar.pack(fill="x")


def on_close():
    print("Application is closing...")
    # Cancel all pending Tkinter callbacks
    try:
        app.after_cancel(app.after_id)  # Cancel any pending callbacks if stored
    except AttributeError:
        pass  # Ignore if no callbacks are pending

    # Close all matplotlib figures
    plt.close('all')
    # Destroy the application window
    app.quit()  # Stop the Tkinter main loop
    app.destroy()  # Destroy the window

# Bind the close event
app.protocol("WM_DELETE_WINDOW", on_close)


# window main loop
app.mainloop()
