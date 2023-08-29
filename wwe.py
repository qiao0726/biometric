import os
from PIL import Image

# Set path to folder containing images
folder_path = "/home/qn/datasets/czrkzp1_5w_subset"
finished = 0

# Loop through all files in folder
for filename in os.listdir(folder_path):
    # Check if file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        # Duplicate image 10 times and save with new names
        for i in range(1, 11):
            new_filename = f"{i}_{filename}"
            new_image_path = os.path.join(folder_path, new_filename)
            image.save(new_image_path)
            finished += 1
            print(f"Finished {finished} images")