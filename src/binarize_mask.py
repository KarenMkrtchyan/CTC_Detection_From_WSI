import os
from pathlib import Path
import cv2
import numpy as np

# Input and output directories
input_dir = Path("./masks")
output_dir = Path("./masks_binary")

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Loop over all PNG images in the input directory
for image_file in input_dir.glob("*.png"):
    # Read the image as a 2D matrix of pixel values
    image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # Binarize the image: set all non-zero values to 255 (white)
    binary_image = np.where(image > 0, 255, 0).astype(np.uint8)

    # Save the binary image to the output directory
    output_path = output_dir / image_file.name
    cv2.imwrite(str(output_path), binary_image)
    print(f"Saved binary mask to {output_path}")