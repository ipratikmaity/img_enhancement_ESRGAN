import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

# Add ESRGAN model path to system path
import sys
sys.path.append(r'D:\6th semester [cse 36]\mini project- 6th sem\ESRGAN-master\ESRGAN-master')

from RRDBNet_arch import RRDBNet

# Directories
input_dir = r'D:\6th semester [cse 36]\mini project- 6th sem\test images- pm\images'
output_dir = r'D:\6th semester [cse 36]\mini project- 6th sem\test images- pm\enhanced_images'
model_path = r'D:\6th semester [cse 36]\mini project- 6th sem\ESRGAN-20250325T075728Z-001\ESRGAN\models\RRDB_ESRGAN_x4.pth'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained ESRGAN model
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model, device

model, device = load_model(model_path)
print("ESRGAN model loaded successfully.")

# Convolution Implementation
def perform_convolution(input_tensor, kernel):
    """Perform convolution with kernel matching the number of input channels."""
    c = input_tensor.size(1)  # Get the number of channels
    kernel = kernel.repeat(c, 1, 1, 1)  # Match the number of channels
    return torch.nn.functional.conv2d(input_tensor, kernel, groups=c)

# Pixel Shuffle Fallback Implementation
def pixel_shuffle_safe(input_tensor, upscale_factor):
    """Safe pixel shuffle that validates input."""
    b, c, h, w = input_tensor.size()
    if c % (upscale_factor ** 2) != 0:
        print(" ")
        return input_tensor  # Return without upscaling
    new_c = c // (upscale_factor ** 2)
    upscale_h, upscale_w = h * upscale_factor, w * upscale_factor
    input_tensor = input_tensor.view(b, new_c, upscale_factor, upscale_factor, h, w)
    input_tensor = input_tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
    return input_tensor.view(b, new_c, upscale_h, upscale_w)

# Process and enhance images
def enhance_images(input_dir, output_dir, model, device):
    # Example kernel for convolution (e.g., 3x3 Laplacian kernel)
    kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32).to(device)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        # Read and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ToTensor()(img).unsqueeze(0).to(device)

        # Apply convolution using the custom kernel (optional)
        convolved_img = perform_convolution(img, kernel)

        # Enhance the image using the ESRGAN model
        with torch.no_grad():
            enhanced_img = model(convolved_img)

        # Apply pixel shuffle for upscaling safely
        enhanced_img = pixel_shuffle_safe(enhanced_img, upscale_factor=4)

        # Post-process and save the enhanced image
        enhanced_img = enhanced_img.squeeze(0).cpu()
        enhanced_img = ToPILImage()(enhanced_img)
        enhanced_img.save(os.path.join(output_dir, img_name))

        print(f"Enhanced {img_name} and saved to {output_dir}.")

# Run the image enhancement
enhance_images(input_dir, output_dir, model, device)
print("All images have been processed and enhanced.")