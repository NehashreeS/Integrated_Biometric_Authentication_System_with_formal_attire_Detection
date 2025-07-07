import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import csv
import numpy as np
import attire_detection  # Import the attire detection module
from finger import submit_image, browse_image
from fingerprint_feature_extraction import extract_minutiae_features
from fingerprint_feature_extraction import MinutiaeFeature
import os
import json
import cv2
import mediapipe as mp
import numpy as np
import time

def analyze_faces_and_save(person_folder_path, output_json):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    # Extract the person's name from the folder name
    person_name = os.path.basename(person_folder_path.strip(os.sep))

    # Prepare a list to store image data for this person
    person_data = []

    # Check if the folder exists
    if not os.path.exists(person_folder_path):
        print(f"Error: The folder '{person_folder_path}' does not exist.")
        return

    # Iterate through all image files in the folder
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)

        # Check if it's a valid image file
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read and process the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_name}")
                continue

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                # Extract landmarks for the first detected face
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [
                    {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                    for landmark in face_landmarks.landmark
                ]
                person_data.append({"image": image_name, "landmarks": landmarks})
            else:
                print(f"No face detected in image: {image_name}")

    # Load existing JSON data if the file exists
    if os.path.exists(output_json):
        with open(output_json, "r") as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {}  # Start with an empty dictionary if the file is empty or corrupted
    else:
        data = {}

    # Add or update the current person's data
    data[person_name] = person_data

    # Save updated data to the JSON file
    with open(output_json, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Analysis complete. Results saved to {output_json}")

def video_to_images( destination_folder, person_name):
    # Create destination folder if it doesn't exist
    person_folder = os.path.join(destination_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the video file.")
        return

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    frame_count = 0
    max_frames = fps * 5  # Process only the first 10 seconds

    while frame_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            print("Video processing completed or reached the end of the video.")
            break

        # Increment frame count
        frame_count += 1

        # Generate filename and save directly to folder
        filename = f"{person_name}_{frame_count:04d}.jpg"
        filepath = os.path.join(person_folder, filename)

        # Save the frame as an image
        cv2.imwrite(filepath, frame)

    # Release the video capture object
    cap.release()
    print(f"Processed and saved {frame_count} frames from the video.")


def add_face(person_name):
    # Define the destination folder (Downloads folder)
    if len(person_name) != 0:
        downloads_folder = os.path.expanduser("~/Downloads")
        # Run the video to image conversion
        video_to_images( downloads_folder, person_name)
        print(f"Images have been saved in the folder: {os.path.join(downloads_folder, person_name)}")
        person_folder_path = f"C:/Users/localadmin/Downloads/{person_name}"
        output_json = r"C:\Users\localadmin\Downloads\person_analysis.json"
        analyze_faces_and_save(person_folder_path, output_json)
    else:
        messagebox.showinfo("Chatbot", "Person name is mandatory")

# Global variables
model = attire_detection.load_model()  # Load the YOLO model once
current_frame = None  # To store the current frame for capturing

def analyze_faces_and_save(person_folder_path, output_json):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    # Prepare a list to store image data
    all_person_data = {}

    # Check if the folder exists
    if not os.path.exists(person_folder_path):
        print(f"Error: The folder '{person_folder_path}' does not exist.")
        return

    # Iterate through all files in the folder
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)

        # Check if it's a valid image file
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(image_name)[0]  # Use image name as person's name
            print(f"Processing image: {image_name} for person: {person_name}")

            # Read and process the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_name}")
                continue

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            person_data = []
            if results.multi_face_landmarks:
                # Extract landmarks for the first detected face
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [
                    {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                    for landmark in face_landmarks.landmark
                ]
                person_data.append({"image": image_name, "landmarks": landmarks})
            else:
                print(f"No face detected in image: {image_name}")

            # Add or update the current person's data (use image name as person's name)
            all_person_data[person_name] = person_data

    # Load existing JSON data if the file exists
    if os.path.exists(output_json):
        with open(output_json, "r") as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {}  # Start with an empty dictionary if the file is empty or corrupted
    else:
        data = {}

    # Merge the new data with the existing data
    data.update(all_person_data)

    # Save updated data to the JSON file
    with open(output_json, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Analysis complete. Results saved to {output_json}")

def get_person_name_from_image(input_image_path, json_file_path, similarity_threshold=0.01):
    # Initialize MediaPipe FaceMesh
    start_time = time.time()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    # Load the existing JSON data
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                print("Error loading JSON data.")
                return None
    else:
        print("JSON file not found.")
        return None

    # Read and process the input image
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error reading image: {input_image_path}")
        return None

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("No face detected in the input image.")
        return None

    # Extract landmarks for the first detected face in the input image
    input_face_landmarks = results.multi_face_landmarks[0]
    input_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in input_face_landmarks.landmark])

    min_distance = float('inf')
    person_name = None

    # Compare the input image's landmarks with stored landmarks
    for stored_person_name, person_data in data.items():
        for person_image_data in person_data:
            stored_landmarks = np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in person_image_data["landmarks"]])

            # Calculate Euclidean distance between input and stored landmarks
            distance = np.sum(np.linalg.norm(input_landmarks - stored_landmarks, axis=1))

            # Calculate similarity as 1 / (1 + distance)
            similarity = 1 / (1 + distance)

            if similarity >= similarity_threshold:
                # If similarity is above the threshold, consider it a match
                if distance < min_distance:
                    min_distance = distance
                    person_name = stored_person_name

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")

    if person_name:
        print(f"The person in the image is: {person_name}")
    else:
        print("No match found.")

    return person_name


def update_camera():
    global cap, canvas, photo, current_frame
    ret, frame = cap.read()

    if ret:
        current_frame = frame  # Store the latest frame for capturing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (screen_width, screen_height))  # Resize to fullscreen
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    root.after(10, update_camera)


def show_chatbot_message():
    popup = tk.Toplevel(root)
    popup.title("ChatBot")
    popup.geometry("200x100")
    tk.Label(popup, text="Hello! How can I help you?").pack(pady=10)
    tk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)

def admin_close(admin_window):
    admin_window.destroy()
    root.deiconify()

def open_admin_page():

    root.withdraw()
    admin_window = tk.Toplevel(root)
    admin_window.title("Admin Page")
    admin_window.geometry("1920x1280")  # Adjust dimensions as needed
    admin_window.grab_set()
    # Left frame
    left_frame = tk.Frame(admin_window, bg="white", width=200, height=600)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    # Right frame
    right_frame = tk.Frame(admin_window, bg="lightgray", width=600, height=600)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Buttons in the left frame
    tk.Button(left_frame, text="Add Profile", font=("Arial", 14), bg="lightblue",
              command=lambda: display_add_profile_form(right_frame)).pack(pady=25, padx=10, fill=tk.X)
   

    tk.Button(left_frame, text="Close", font=("Arial", 12), bg="red", fg="white", command=lambda:admin_close(admin_window)).pack(pady=20)


def display_add_profile_form(right_frame):
    for widget in right_frame.winfo_children():
        widget.destroy()  # Clear the frame

    # Create form fields
    tk.Label(right_frame, text="Add New Profile", font=("Arial", 18), bg="lightgray").pack(pady=10)


    tk.Label(right_frame, text="Name:", font=("Arial", 14), bg="lightgray").pack(pady=5)
    name_entry = tk.Entry(right_frame, font=("Arial", 12))
    name_entry.pack(pady=5)

    video_path = os.path.expanduser("~/Downloads")
    downloads_folder = os.path.expanduser("~/Downloads")
   
    tk.Button(right_frame, text="Upload Face Video", font=("Arial", 12),
              command=lambda:add_face(name_entry.get())).pack(pady=20) #check it is correct?


    tk.Label(right_frame, text="Fingerprint Image:", font=("Arial", 14), bg="lightgray").pack(pady=5)
    fingerprint_image_path = tk.StringVar()
    tk.Entry(right_frame, textvariable=fingerprint_image_path, font=("Arial", 12), state="readonly").pack(pady=5)
    tk.Button(right_frame, text="Upload Fingerprint Image", font=("Arial", 12),
              command=lambda: select_fifile(name_entry.get(),fingerprint_image_path)).pack(pady=5)

    # Add User Button
    tk.Button(right_frame, text="Add User", font=("Arial", 14), bg="green", fg="white",
              command=lambda: add_user( name_entry.get(),  fingerprint_image_path.get(),right_frame)).pack(pady=20)





def select_file(path_variable):
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
        title="Select an Image"
    )
    if file_path:
        path_variable.set(file_path)

def save_minutiae_to_npy2(FeaturesTerm, FeaturesBif, file_id, file_extension='minutiae_features.npy'):
    """
    Saves the Termination and Bifurcation minutiae features to a .npy file with a dynamic filename.

    Parameters:
    - FeaturesTerm: List of Termination minutiae features
    - FeaturesBif: List of Bifurcation minutiae features
    - file_id: The unique identifier to include in the filename
    - file_extension: The suffix for the file (default is 'minutiae_features.npy')
    """
    # Create a structured numpy array for storing minutiae features
    dtype = [('location_x', 'i4'), ('location_y', 'i4'), ('orientation', 'O'), ('minutiae_type', 'U10')]

    # Prepare the data in a list of tuples
    minutiae_data = []

    # Add Termination minutiae to the data list
    for feature in FeaturesTerm:
        minutiae_data.append((feature.locX, feature.locY, feature.Orientation, 'Termination'))

    # Add Bifurcation minutiae to the data list
    for feature in FeaturesBif:
        minutiae_data.append((feature.locX, feature.locY, feature.Orientation, 'Bifurcation'))

    # Convert the list to a structured numpy array
    minutiae_array = np.array(minutiae_data, dtype=dtype)

    save_directory = "C:/Users/localadmin/Desktop/uuuu/fingerprint_scans"
    #save npy to database
    # Ensure the database directory exists
    os.makedirs(save_directory, exist_ok=True)

    filename = os.path.join(save_directory, f"{file_id}_{file_extension}")

    
    # Save the array to a .npy file
    np.save(filename, minutiae_array)

    print(f"Minutiae features saved in {file_id}_{file_extension} format.")

def select_fifile(name_entry,path_variable):

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg:*.BMP")],
        title="Select an Image"
    )
    if file_path:
        path_variable.set(file_path)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    '''
    def get_id_from_filepath(filepath):
        # Extract the filename from the full file path
        file_name = os.path.basename(filepath)  # This gets the last part of the path, i.e., the filename
        # Split the filename at the first underscore and take the first part
        file_id = file_name.split('_')[0]  # Take the part before the first underscore
        return file_id'''
    
    file_id = name_entry

    # Check if the image is loaded correctly
    if img is None:
        raise ValueError("The image could not be loaded. Please check the image path.")

    # Enhance the image using histogram equalization
    img_eq = cv2.equalizeHist(img)

    # Check if the enhanced image is empty
    if img_eq is None or img_eq.size == 0:
        raise ValueError("The enhanced image is empty. There may be an issue with the enhancement process.")

    # Apply Gabor filtering to enhance ridges
    def gabor_filtering(img):
        ksize = 21
        sigma = 5.0
        theta = np.pi / 4
        lambd = 10.0
        gamma = 0.5
        gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)

        # Check if the Gabor kernel is valid
        if gabor_kernel is None or gabor_kernel.size == 0:
            raise ValueError("Failed to create Gabor kernel.")

        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_kernel)
        return filtered_img

    # Apply Gabor filtering to the enhanced image
    img_gabor = gabor_filtering(img_eq)



    # Binarize the image
    _, img_binary = cv2.threshold(img_gabor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Now, extract minutiae features
    FeaturesTerm, FeaturesBif = extract_minutiae_features(
        img_binary,
        spuriousMinutiaeThresh=10,
        invertImage=False,
        showResult=False,
        saveResult=False
    )

    save_minutiae_to_npy2(FeaturesTerm, FeaturesBif, file_id)


def add_user( name, fingerprint_image,right_frame):
    if not   name  or not fingerprint_image:
        messagebox.showerror("Error", "All fields are required!")
        return

    # Save user information (can be extended to save to a database)
    user_info = f" Name: {name}\nFingerprint Image: {fingerprint_image}"
    print(user_info)

    messagebox.showinfo("Success", f"User {name} added successfully!")
    right_frame.destroy()



def menu_action():
    menu_page = tk.Frame(root, bg="white", width=300, height=screen_height)
    menu_page.place(x=-300, y=0)

    tk.Label(menu_page, text="Menu Options", font=("Arial", 18), bg="white").pack(pady=20)
    tk.Button(menu_page, text="Add Profile ", font=("Arial", 14), bg="white", command=open_admin_page).pack(pady=10, anchor=tk.W, padx=20)
    tk.Button(menu_page, text="FingerPrint", font=("Arial", 14), bg="white",command=create_gui).pack(pady=10, anchor=tk.W, padx=20)
    
    close_btn = tk.Button(menu_page, text="Close", command=lambda: glide_menu(menu_page, direction=-1), bg="red", fg="white")
    close_btn.pack(pady=10)

    glide_menu(menu_page, direction=1)


def glide_menu(menu_page, direction=1):
    current_x = menu_page.winfo_x()
    target_x = 0 if direction == 1 else -300
    step = 10 if direction == 1 else -10

    def animate():
        nonlocal current_x
        current_x += step
        menu_page.place(x=current_x, y=0)
        if (direction == 1 and current_x < target_x) or (direction == -1 and current_x > target_x):
            menu_page.after(10, animate)
        elif direction == -1:
            menu_page.destroy()

    animate()


def capture_photo():
    global current_frame, model  # Use the globally loaded model
    if current_frame is not None:
        # Get the Downloads folder path
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(downloads_folder):
            os.makedirs(downloads_folder)

        # Save the captured image
        image_path = os.path.join(downloads_folder, "captured_image.jpg")
        output_json = r"C:\Users\localadmin\Downloads\person_analysis.json" 
        cv2.imwrite(image_path, current_frame)
        print(f"Photo saved at: {image_path}")

        # Perform attire detection
        try:
            person_name = get_person_name_from_image(image_path, output_json)
            if person_name is not None:
                result = attire_detection.predict_attire(image_path, model)
                if result=="Unknown":
                    messagebox.showinfo("ChatBot","Please stand in correct position and press capture again")
                else:
                    print(f"Attire Prediction: {result}")
                    messagebox.showinfo("ChatBot", f"{person_name} is wearing: {result}")
            else:
                messagebox.showerror("Error", f"sorry face detection is not possible: {str(e)}. Try fingerprint scanning instead.")
        except Exception as e:
            messagebox.showerror("Error", f"Attire is not visible,Please press capture: {str(e)}")
    else:
        messagebox.showerror("Error", "No frame captured. Please try again.")

def create_gui():
    f_window = tk.Toplevel(root)
    f_window.title("Fingerprint Scanner")
    f_window.grab_set()

    # Set window size and remove the default title bar (makes it look like a pop-up)
    f_window.geometry("700x300")  # Fixed size for the pop-up window
    f_window.overrideredirect(True)  # Remove the window's title bar to make it borderless

    # Get screen width and height
    screen_width = f_window.winfo_screenwidth()
    screen_height = f_window.winfo_screenheight()

    # Position the window at the center of the screen
    window_width = 700
    window_height = 300
    position_top = (screen_height // 2) - (window_height // 2)
    position_left = (screen_width // 2) - (window_width // 2)
    f_window.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

    # Create the main frame that will be the size of the pop-up box
    main_frame = tk.Frame(f_window, bg="white", highlightbackground="black", highlightthickness=2)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Add the title label for the box
    title_label = tk.Label(main_frame, text="Fingerprint Scanner", bg="lightblue", font=("Helvetica", 24, "bold"), anchor=tk.CENTER, padx=5, pady=5)
    title_label.pack(fill=tk.X)

    # Adjust padding to separate title from other elements
    content_frame = tk.Frame(main_frame, bg="white", padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)


    # Create the label and entry for Image Filename
    image_label = tk.Label(content_frame, text="Image Filename", bg="white", font=("Helvetica", 20))
    image_label.grid(row=10, column=0, sticky=tk.W, padx=(0, 10), pady=(30, 10))  
    image_entry = ttk.Entry(content_frame, width=30, font=("Helvetica", 16))
    image_entry.grid(row=10, column=3, pady=(30, 10))  

    # Create a Browse button to allow the user to select an image file
    browse_button = ttk.Button(content_frame, text="Browse", command=lambda: browse_image(image_entry))
    browse_button.grid(row=10, column=4, pady=(30, 10))  

    # Create a Submit button to move the selected image to the folder
    submit_button = tk.Button(main_frame, text="Submit", command=lambda: submit_image(image_entry,f_window), font=("Helvetica", 16, "bold"), bg="white", fg="black", relief="raised", width=10, height=3)
    submit_button.pack(side=tk.BOTTOM, pady=20, padx=20)

    # Function to close the window when clicking the X in the top-right corner
    def close_window():
        f_window.destroy()


    # Close button (X) at the top-right corner inside main_frame
    close_button = tk.Button(main_frame, text="X", command=close_window, bg="red", fg="white", font=("Helvetica", 19), relief="flat", width=3, height=1)
    close_button.place(relx=1.0, rely=0.0, anchor="ne")  # Positioned in the top-right corner inside main_frame




# Initialize the application window
root = tk.Tk()
root.title("Live Camera Feed with Chatbot")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.geometry(f"{screen_width}x{screen_height}")
root.attributes("-fullscreen", True)

canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="black")
canvas.pack(fill=tk.BOTH, expand=True)

chatbot_img = ImageTk.PhotoImage(Image.open("chatbot.png").resize((100, 100)))
chatbot_button = tk.Button(root, image=chatbot_img, command=show_chatbot_message, borderwidth=0)
chatbot_button.place(relx=0.95, rely=0.85, anchor=tk.SE)

menu_button = tk.Button(root, text="≡", command=menu_action, font=("Arial", 20), width=5, height=2)
menu_button.place(relx=0.02, rely=0.02, anchor=tk.NW)

# Capture button
capture_button = tk.Button(root, text="Capture", command=capture_photo, font=("Arial", 16), bg="green", fg="white", width=10)
capture_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

cap = cv2.VideoCapture(0)
current_frame = None  # To store the current frame for capturing

update_camera()

root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

root.mainloop()
