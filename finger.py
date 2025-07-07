import tkinter as tk
from tkinter import ttk
from tkinter import filedialog,messagebox
import os
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, draw
from fingerprint_feature_extraction import extract_minutiae_features
from fingerprint_feature_extraction import MinutiaeFeature

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_npy(query_file, folder_path):
    """
    Compares a given .npy file with other .npy files in the folder and returns the folder name before '_'.
    
    Parameters:
    - query_file: Path of the query .npy file.
    - folder_path: Path of the folder containing stored .npy files.
    
    Returns:
    - The name of the folder that best matches the query file (before '_').
    """
    # Load query minutiae data
    query_data = np.load(query_file, allow_pickle=True)

    # Extract numeric features for comparison (ignore 'orientation' and 'minutiae_type')
    query_numeric = np.array([(x['location_x'], x['location_y']) for x in query_data])

    best_match = None
    lowest_score = float('inf')

    # Iterate through all .npy files in the folder
    for stored_file in os.listdir(folder_path):
        if stored_file.endswith('.npy'):
            stored_path = os.path.join(folder_path, stored_file)
            stored_data = np.load(stored_path, allow_pickle=True)

            # Extract numeric features
            stored_numeric = np.array([(x['location_x'], x['location_y']) for x in stored_data])

            # Compute similarity score (sum of absolute differences)
            if query_numeric.shape == stored_numeric.shape:  # Ensure shapes match
                score = np.sum(np.abs(query_numeric - stored_numeric))
            else:
                score = float('inf')  # If shapes don't match, assign a high difference

            # Update best match if the new score is lower
            if score < lowest_score:
                lowest_score = score
                best_match = stored_file.split('_')[0]  # Extract folder name before '_'
    

    print("Best match found in folder:", best_match)
    
    return best_match if best_match else "No match found"



# Example usage

def save_minutiae_to_npy1(FeaturesTerm, FeaturesBif, file_id, file_extension='minutiae_features.npy'):
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

    save_directory = "C:/Users/localadmin/Desktop/Finger Test"

    # Ensure the database directory exists
    os.makedirs(save_directory, exist_ok=True)

    filename = os.path.join(save_directory, f"{file_id}_{file_extension}")
    

    
    # Save the array to a .npy file
    np.save(filename, minutiae_array)
    folder_path = "C:/Users/localadmin/Desktop/uuuu/fingerprint_scans"
    best_match = find_similar_npy(filename, folder_path)
    print(f"Minutiae features saved in {file_id}_{file_extension} format.")
    if best_match != "No match found":
        messagebox.showinfo("Authentication", f"{best_match} is authenticated!")
    else:
        messagebox.showerror("Authentication Failed", "No match found. Authentication failed.")

# Function to browse and select the image
def browse_image(image_entry):
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.BMP;*.gif")])
    
    if file_path:
        # Display the selected file path in the image entry field
        image_entry.delete(0, tk.END)
        image_entry.insert(0, file_path)

# Function to move the selected image to a folder
def submit_image(image_entry,f_window):
    file_path = image_entry.get()
    if file_path and os.path.exists(file_path):
        # Create the destination folder if it doesn't exist
        folder_path = "submitted_images"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
# Load the fingerprint image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def get_id_from_filepath(filepath):
        # Extract the filename from the full file path
        file_name = os.path.basename(filepath)  # This gets the last part of the path, i.e., the filename
        # Split the filename at the first underscore and take the first part
        file_id = file_name.split('_')[0]  # Take the part before the first underscore
        return file_id
    
    f_window.destroy()

    file_id = get_id_from_filepath(file_path)

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

    
    save_minutiae_to_npy1(FeaturesTerm, FeaturesBif, file_id)

    
    

    # Show authentication message
    

    # Get the file name from the path
        #file_name = os.path.basename(file_path)
        
        # Create the destination file path
        #destination_path = os.path.join(folder_path, file_name)
        
        #try:
            # Move the file to the destination folder
            #shutil.move(file_path, destination_path)
            #print(f"Image moved to {destination_path}")
        #except Exception as e:
            #print(f"Error: {e}")
    #else:
        #print("Please select a valid image file.")'''

