import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from sklearn.cluster import KMeans
import numpy as np
def show_image(img_array, parent, title="Image"):
    img = Image.fromarray(img_array)
    img = img.resize((300,200))  # Resize for better GUI fit
    imgtk = ImageTk.PhotoImage(img)

    frame = tk.LabelFrame(parent, text=title, font=("Arial", 12, "bold"), bg="#ffffff", fg="#333333", bd=30, relief="ridge")
    frame.pack(side="left", padx=15, pady=15)

    label = tk.Label(frame, image=imgtk, bg="#ffffff")
    label.image = imgtk  # Keep reference
    label.pack()

def import_pic():
    # Clear previous images
    for widget in inner_frame.winfo_children():
        widget.destroy()

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        messagebox.showwarning("No file selected", "Please select an image file.")
        return

    # Get the user input for k
    try:
        k = int(k_entry.get())
        if k <= 0 or k > 255:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a positive integer for number of clusters.")
        loading_label.config(text="")
        return

    # Show loading label
    loading_label.config(text="Loading, please wait...")
    root.update_idletasks()

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display original image
    show_image(image, inner_frame, title="Original Image")

    # Reshape and apply KMeans
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    labels = kmeans.labels_   # cluster id
    centers = kmeans.cluster_centers_  # RGB values of clusters

    # Display each cluster separately
    for cluster_num in range(k):
        mask = (labels == cluster_num).reshape(image.shape[:2])
        cluster_img = np.zeros_like(image)
        cluster_img[mask] = centers[cluster_num].astype(np.uint8) #converts the float values (e.g., 128.5) to integers (e.g., 128), which are valid image pixel values.

        show_image(cluster_img, inner_frame, title=f'Cluster {cluster_num + 1}')

    # Hide loading label
    loading_label.config(text="")

    # Update scroll region
    images_canvas.update_idletasks()
    images_canvas.config(scrollregion=images_canvas.bbox("all"))

def on_mousewheel(event):
    # Scroll canvas left/right with smoother speed
    images_canvas.xview_scroll(int(-1 * (event.delta /60 )), "units")  # Smoother scrolling

# Main GUI window
root = tk.Tk()
root.title("Image Clustering GUI")
root.geometry("1600x900")

# Plain background color
root.configure(bg="#1e1e2f")

# Title label
title_label = tk.Label(root, text="Image color segmentation", font=("Helvetica", 26, "bold"), bg="#1e1e2f", fg="#00d4ff")
title_label.pack(pady=20)

# Import button
import_button = tk.Button(root, text="Import Picture", command=import_pic,
                          font=("Arial", 16, "bold"), bg="#00d4ff", fg="#1e1e2f",
                          activebackground="#0099cc", activeforeground="white",
                          relief="ridge", bd=4)
import_button.pack(pady=10)

# Entry for K value
k_frame = tk.Frame(root, bg="#1e1e2f")
k_frame.pack(pady=10)

k_label = tk.Label(k_frame, text="Number of Colors (k):", font=("Arial", 14), bg="#1e1e2f", fg="#00ffaa")
k_label.pack(side="left", padx=5)

k_entry = tk.Entry(k_frame, font=("Arial", 14), width=5)
k_entry.pack(side="left")
k_entry.insert(0, "5")  # Default value

# Loading label
loading_label = tk.Label(root, text="", font=("Arial", 14, "italic"), bg="#1e1e2f", fg="#00ffaa")
loading_label.pack()

# Canvas for images with scrollbar
images_canvas = tk.Canvas(root, bg="#1e1e2f", height=500, highlightthickness=0)
images_canvas.pack(fill="both", expand=True, padx=20, pady=20)

# Horizontal scrollbar
h_scroll = tk.Scrollbar(root, orient="horizontal", command=images_canvas.xview)
h_scroll.pack(fill="x")
images_canvas.configure(xscrollcommand=h_scroll.set)

# Inner frame inside canvas
inner_frame = tk.Frame(images_canvas, bg="#1e1e2f")
images_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Bind mousewheel for smooth scroll
images_canvas.bind_all("<MouseWheel>", on_mousewheel)

# Start GUI
root.mainloop()
