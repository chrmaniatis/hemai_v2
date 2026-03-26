import os
import cv2
import pandas as pd
import numpy as np
import shutil
from cellpose import models, io, plot
from typing import Union, Tuple, Dict
import time

def crop_black_borders(img, threshold=40):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find non-black areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find all non-zero (non-black) pixel coordinates
    coords = cv2.findNonZero(mask)

    # Get bounding box of the non-black area
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the original image using the bounding box
    cropped = img[y+10:y+h-10, x+10:x+w-10]

    return cropped
    
# Input and output folders
input_folder = "/content/MyDrive/MyDrive/Aima_pipeline/Images"
move_folder = "/content/MyDrive/MyDrive/Aima_pipeline/Images/prc"
output_folder = "/content/MyDrive/MyDrive/Aima_pipeline/Images/cropped"


os.makedirs(output_folder, exist_ok=True)
os.makedirs(move_folder, exist_ok=True)

start = time.time()
# Load Cellpose model
model = models.CellposeModel(gpu=True)

# List jpg files
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg") or f.lower().endswith(".png")  or f.lower().endswith(".tif")]
for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)

    img = io.imread(img_path)
    cropped_img = crop_black_borders(img)
    masks, flows, styles = model.eval(cropped_img, diameter=50,flow_threshold=1.0,  cellprob_threshold=-3.0, min_size = 200)

    num_cells = masks.max()

    # Create a unique subdirectory for this image
    base_name = os.path.splitext(img_file)[0]
    image_output_folder = os.path.join(output_folder, base_name)
    os.makedirs(image_output_folder, exist_ok=True)

    # Extract cells and features
    for cell_id in range(1, num_cells + 1):
        cell_mask = (masks == cell_id)

        # Find bounding box
        y_indices, x_indices = cell_mask.nonzero()
        y1, y2 = y_indices.min(), y_indices.max()
        x1, x2 = x_indices.min(), x_indices.max()

        # Crop cell
        cropped_cell = cropped_img[y1:y2, x1:x2].copy()

        if cropped_cell.dtype != 'uint8':
            cropped_cell = (cropped_cell * 255).astype('uint8')

        if len(cropped_cell.shape) == 3 and cropped_cell.shape[2] == 3:
            cropped_cell = cv2.cvtColor(cropped_cell, cv2.COLOR_RGB2BGR)

        save_name = f"{base_name}_cell{cell_id}.jpg"
        save_path = os.path.join(image_output_folder, save_name)
        cv2.imwrite(save_path, cropped_cell)

    print(f"Processed {img_file}: extracted {num_cells} cells.")
    shutil.move(img_path, os.path.join(move_folder, img_file))

print("✅ Done! Cropped cells are saved in", output_folder)
elapsed = time.time() - start
hours, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
print(f"✅ Script finished in {int(hours):02d}h {int(mins):02d}m {secs:.2f}s")
