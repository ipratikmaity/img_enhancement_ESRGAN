
import cv2
import numpy as np
import os


# Sharpening function using a more aggressive kernel
def sharpen_image(image, alpha=0.8):
    # Increase the intensity of the sharpening kernel
    kernel = np.array([[0, -6, 0],
                       [-5, 24, -5],
                       [0, -7, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)  # Apply sharpening kernel
    
    # Blend the original and sharpened image for more control
    blended = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
    return blended

# Paths for input/output
blurred_folder = r"D:\6th semester [cse 36]\mini project- 6th sem\test images- pm\images"  # Folder containing blurred images
output_folder = r"D:\6th semester [cse 36]\mini project- 6th sem\test images- pm\sharpened_images"  # Folder to save sharpened images
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Process and sharpen each image
for image_name in os.listdir(blurred_folder):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):
        img_path = os.path.join(blurred_folder, image_name)
        image = cv2.imread(img_path)  # Load the image
        sharpened = sharpen_image(image, alpha=0.8)  # Apply stronger sharpening with blending
        
        # Save the sharpened image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, sharpened)

print("Sharper and blended images saved to:", output_folder)