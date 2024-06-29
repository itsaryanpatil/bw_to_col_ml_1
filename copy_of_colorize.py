

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#Importing Libraries

#	•	torch: The core library for PyTorch, used for tensor operations and neural network building.
#	•	torch.nn: Contains modules and classes for building neural network layers.
#	•	torch.optim: Provides optimization algorithms like SGD, Adam, etc.
#	•	torchvision: A package containing popular datasets, model architectures, and image transformations.
#	•	torchvision.transforms: Utilities for data transformation, such as converting images to tensors, normalizing, resizing, etc.

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))
else:
  print("No NVIDIA driver found. Using CPU")

#Check and Print GPU Information:
	#•	torch.cuda.is_available(): Checks if a CUDA-capable GPU is available.
	#•	torch.cuda.current_device(): Returns the index of the currently selected GPU.
	#•	torch.cuda.device(0): Returns a device object for the GPU at index 0.
	#•	torch.cuda.device_count(): Returns the number of GPUs available.
	#•	torch.cuda.get_device_name(0): Returns the name of the GPU at index 0.

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

#Loading and Preprocessing the CIFAR-10 Dataset

#Transformations

	#•	ToTensor(): This transformation converts the PIL images (loaded from the dataset) to PyTorch tensors. The image data is scaled from a range of [0, 255] to [0, 1].

#Loading the Dataset

	#•	train_dataset: This loads the CIFAR-10 training dataset with the specified transformation.
	#•	train_loader: This wraps the dataset in a DataLoader, which allows for easy batching, shuffling, and loading of the data in parallel using multiple worker threads.
	#•	test_dataset: Similarly, this loads the CIFAR-10 test dataset with the same transformation.
	#•	test_loader: This wraps the test dataset in a DataLoader.

# Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x
#This ColorizationNet class defines a simple convolutional neural network (CNN) architecture for colorization. Let’s go through it step by step:

#Model Architecture

#__init__ Method

#	•	Convolutional Layers:
#	•	self.conv1: Takes an input of 1 channel (grayscale) and outputs 64 channels, using a kernel size of 5x5, with a stride of 1, padding of 4, and dilation of 2.
#	•	self.conv2: Continues from 64 channels to another 64 channels, with similar parameters.
#	•	self.conv3: Increases the output channels to 128, maintaining the same kernel size, stride, padding, and dilation.
#	•	self.conv4: Outputs 3 channels, which correspond to the RGB color channels, using a kernel size of 5x5, stride of 1, padding of 4, and dilation of 2.

#forward Method

#	•	Activation Functions:
#	•	nn.functional.relu: Applies ReLU activation after each convolutional layer, except for the last layer.
#	•	torch.sigmoid: Applies sigmoid activation to the output of self.conv4. This function scales the output to the range [0, 1], which is suitable for representing color channels.

#Explanation

#	•	ReLU Activation: Introduces non-linearity after each convolutional layer, allowing the network to learn complex patterns.
#	•	Sigmoid Activation: Squashes the output values to the range [0, 1], suitable for representing color values where each pixel’s RGB components range from 0 (black) to 1 (white).

model = ColorizationNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert RGB image to grayscale
def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)

#Model Instantiation

#	•	model = ColorizationNet().to(device): Creates an instance of your ColorizationNet model and moves it to the specified device (GPU if available, otherwise CPU).

#Loss and Optimizer

#	•	criterion = nn.MSELoss(): Defines Mean Squared Error loss, which is commonly used for regression tasks like image colorization, where the goal is to minimize the difference between predicted and actual pixel values.
#	•	optimizer = optim.Adam(model.parameters(), lr=0.001): Initializes the Adam optimizer, which is efficient for training deep neural networks. It optimizes the parameters of the model (model.parameters()) with a learning rate (lr) set to 0.001.

#rgb_to_gray Function

#	•	rgb_to_gray(img): Converts RGB images to grayscale by taking the mean across the RGB channels (dim=1), resulting in a single-channel grayscale image (keepdim=True keeps the channel dimension).

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        grayscale_images = rgb_to_gray(images).to(device)
        images = images.to(device)

        # Forward pass
        outputs = model(grayscale_images)
        loss = criterion(outputs, images)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print statistics
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Finished Training")

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # Convert from Tensor image and display
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if len(img.shape) == 2:  # grayscale image
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualize_all_three(original_images, grayscale_images, colorized_images, n=5):
    """
    Display grayscale, colorized, and original images side by side.
    n: number of images to display from the batch
    """
    fig = plt.figure(figsize=(3*n, 4))
    for i in range(n):
        # Display original image
        ax = plt.subplot(1, 3*n, 3*i + 1)
        imshow(original_images[i])
        ax.set_title("Original")
        ax.axis("off")

        # Display original grayscale image
        ax = plt.subplot(1, 3*n, 3*i + 2)
        imshow(grayscale_images[i])
        ax.set_title("Grayscale")
        ax.axis("off")

        # Display colorized image
        ax = plt.subplot(1, 3*n, 3*i + 3)
        imshow(colorized_images[i])
        ax.set_title("Colorized")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def torch_rgb_to_hsv(rgb):
    """
    Convert an RGB image tensor to HSV.

    Parameters:
    - rgb: tensor of shape (batch_size, 3, height, width) in RGB format in the range [0, 1].

    Returns:
    - hsv: tensor of same shape in HSV format in the range [0, 1].
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val

    # Compute H
    h = torch.zeros_like(r)
    mask = (max_val == r) & (g >= b)
    h[mask] = (g[mask] - b[mask]) / diff[mask]
    mask = (max_val == r) & (g < b)
    h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
    mask = max_val == g
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
    mask = max_val == b
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
    h = h / 6.0
    h[diff == 0.0] = 0.0

    # Compute S
    s = torch.zeros_like(r)
    s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]

    # V is just max_val
    v = max_val

    return torch.stack([h, s, v], dim=1)


def torch_hsv_to_rgb(hsv):
    """
    Convert an HSV image tensor to RGB.

    Parameters:
    - hsv: tensor of shape (batch_size, 3, height, width) in HSV format in the range [0, 1].

    Returns:
    - rgb: tensor of same shape in RGB format in the range [0, 1].
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    r[i_mod == 0.0] = v[i_mod == 0.0]
    g[i_mod == 0.0] = t[i_mod == 0.0]
    b[i_mod == 0.0] = p[i_mod == 0.0]

    r[i_mod == 1.0] = q[i_mod == 1.0]
    g[i_mod == 1.0] = v[i_mod == 1.0]
    b[i_mod == 1.0] = p[i_mod == 1.0]

    r[i_mod == 2.0] = p[i_mod == 2.0]
    g[i_mod == 2.0] = v[i_mod == 2.0]
    b[i_mod == 2.0] = t[i_mod == 2.0]

    r[i_mod == 3.0] = p[i_mod == 3.0]
    g[i_mod == 3.0] = q[i_mod == 3.0]
    b[i_mod == 3.0] = v[i_mod == 3.0]

    r[i_mod == 4.0] = t[i_mod == 4.0]
    g[i_mod == 4.0] = p[i_mod == 4.0]
    b[i_mod == 4.0] = v[i_mod == 4.0]

    r[i_mod == 5.0] = v[i_mod == 5.0]
    g[i_mod == 5.0] = p[i_mod == 5.0]
    b[i_mod == 5.0] = q[i_mod == 5.0]

    return torch.stack([r, g, b], dim=1)

def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    """
    Exaggerate the colors of RGB images.

    Parameters:
    - images: tensor of shape (batch_size, 3, height, width) in RGB format.
    - saturation_factor: factor by which to increase the saturation. Default is 1.5.
    - value_factor: factor by which to increase the value/brightness. Default is 1.2.

    Returns:
    - color_exaggerated_images: tensor of same shape as input, with exaggerated colors.
    """
    # Convert images to the range [0, 1]
    images = (images + 1) / 2.0

    # Convert RGB images to HSV
    images_hsv = torch_rgb_to_hsv(images)

    # Increase the saturation and value components
    images_hsv[:, 1, :, :] = torch.clamp(images_hsv[:, 1, :, :] * saturation_factor, 0, 1)
    images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)

    # Convert the modified HSV images back to RGB
    color_exaggerated_images = torch_hsv_to_rgb(images_hsv)

    # Convert images back to the range [-1, 1]
    color_exaggerated_images = color_exaggerated_images * 2.0 - 1.0

    return color_exaggerated_images

with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        grayscale_images = rgb_to_gray(images).to(device)
        colorized_images = model(grayscale_images)

        # Convert the tensors back to CPU for visualization
        grayscale_images_cpu = grayscale_images.cpu().squeeze(1)  # remove the color channel
        colorized_images_cpu = colorized_images.cpu()
        original_images_cpu = images.cpu()

        #colorized_images_cpu=scale_predicted_colors(colorized_images_cpu)
        colorized_images_cpu=exaggerate_colors(colorized_images_cpu)

        # Visualize the grayscale, colorized, and original images
        visualize_all_three(original_images_cpu, grayscale_images_cpu, colorized_images_cpu)

        if i == 10:  # only do this for up to certain batch for demonstration purposes
            break

from PIL import Image

# Open the image. (Keep your image in the current directory. In my case, the image was horse.jpg)
img = Image.open("/bw/1de.jpg")

# Convert the image to grayscale
gray_img = img.convert("L")

import torchvision.transforms as transforms

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # If you need to normalize, uncomment the following line
    # transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming you want to normalize to [-1, 1] range
])

'''
	1.	transforms.ToTensor():
	•	Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	2.	transforms.Normalize(mean=[0.5], std=[0.5]) (commented out):
	•	Normalizes each channel of the input image. Given mean and std deviations for each channel, it normalizes the image to the range [-1, 1]. This is useful for models that expect inputs to be normalized in this way.
  '''

# Apply the transformations
img_tensor = transform(gray_img).unsqueeze(0)  # Add a batch dimension

# Ensure the model is in evaluation mode
model.eval()

# Move the image tensor to the device where your model is (likely 'cuda' if using GPU)
img_tensor = img_tensor.to(device)

# Get the model's output
with torch.no_grad():
    colorized_tensor = model(img_tensor)

'''
	•	transform(gray_img): Applies the transformation pipeline (transform) defined earlier to gray_img, converting it into a tensor and normalizing it if specified.
	•	.unsqueeze(0): Adds a batch dimension to the tensor. This is necessary because most models expect input tensors in batched format (batch_size, channels, height, width).
  	•	model.eval(): Sets the model (model) to evaluation mode. This is important as it disables dropout layers (if any) and changes the behavior of batch normalization layers during inference.

	•	.to(device): Moves the img_tensor (grayscale image tensor transformed into the desired format) to the specified device (GPU or CPU).
  	•	with torch.no_grad():: Temporarily sets all requires_grad flags to False inside the with block. This disables gradient computation, which is useful during inference to reduce memory usage and speed up computations.
	•	model(img_tensor): Passes the img_tensor through your colorization model (model) to obtain colorized_tensor, which contains the model’s output.
'''

# Convert the tensor back to an image
colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())


# Specify the path where you want to save the model
model_path = 'model.pth'

# Save the model state dictionary
torch.save(model.state_dict(), model_path)

# Plotting the original, grayscale, and colorized images side-by-side
fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure with 1 row and 3 columns

# Display original color image
ax[0].imshow(img)
ax[0].set_title("Original Color Image")
ax[0].axis('off')  # Hide axes

# Display grayscale image
ax[1].imshow(gray_img, cmap='gray')  # Since it's grayscale, use cmap='gray'
ax[1].set_title("Grayscale Image")
ax[1].axis('off')  # Hide axes

# Display colorized image
ax[2].imshow(colorized_img)
ax[2].set_title("Colorized Image")
ax[2].axis('off')  # Hide axes

plt.tight_layout()  # Adjust spacing
plt.show()

gray_img.save("./horse_gre.jpg")

