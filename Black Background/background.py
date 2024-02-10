import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models.segmentation as segmentation
from scipy.ndimage import binary_dilation, binary_erosion
import torchvision.models as models

def load_model(device='cuda'):
    model = models.resnet50(weights=True)
    model.to(device)
    model.eval()
    return model

def process_image(image_path, model, device='cuda'):
    # Load and preprocess the input image
    img = Image.open(image_path).convert("RGB")
    
    # Define the target image size for the model
    target_size = 224

    # Resize the image for the model
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Use the image classification model to get the predicted class label
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # Create a binary mask for the object by selecting pixels of the predicted class
    object_mask = torch.tensor(np.array(predicted_class[0] == predicted_class)).to(device)

    # Create a new image with a black background and the object from the original image
    new_image = Image.new("RGB", img.size, color=(0, 0, 0))
    object_pixels = img.copy().convert("RGBA").crop((0, 0, img.size[0], img.size[1]))
    new_image.paste(object_pixels, (0, 0), mask=Image.fromarray(object_mask.cpu().numpy()))

    return new_image

def process_images_in_folder(input_folder, output_folder, model, device='cuda'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image = process_image(input_image_path, model, device)

        # Save the processed image in the output folder with the same filename
        output_image_path = os.path.join(output_folder, image_file)
        output_image.save(output_image_path)

        print("Processed image saved at:", output_image_path)

if __name__ == "__main__":
    # Check if GPU is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the pre-trained model
    model = load_model(device)

    # Set input and output folder paths
    input_folder = "input"
    output_folder = "output"

    # Process all images in the input folder and save the processed images to the output folder
    process_images_in_folder(input_folder, output_folder, model, device)
