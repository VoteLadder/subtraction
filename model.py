import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import cv2

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust channels if needed for shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Adding the input to the output
        out = self.relu(out)

        return out

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder1 = self._make_encoder_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._make_encoder_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = self._make_encoder_block(512, 1024)  # New encoder level
        self.pool5 = nn.MaxPool2d(2)  # New pool layer

        self.middle = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # New upsampling layer
        self.att5 = AttentionGate(F_g=1024, F_l=1024, F_int=512)  # New attention gate
        self.decoder5 = self._make_decoder_block(2048, 1024)  # New decoder block

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self._make_decoder_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)  # New attention gate
        self.decoder3 = self._make_decoder_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)  # New attention gate
        self.decoder2 = self._make_decoder_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)  # New attention gate
        self.decoder1 = self._make_decoder_block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _make_encoder_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

    def _make_decoder_block(self, in_channels, out_channels):
        return ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        e5 = self.encoder5(self.pool4(e4))  # New encoder level
        
        m = self.middle(self.pool5(e5))  # Updated middle layer
        
        d5 = self.up5(m)
        a5 = self.att5(d5, e5)
        d5 = self.decoder5(torch.cat([d5, a5], dim=1))

        d4 = self.up4(d5)
        a4 = self.att4(d4, e4)
        d4 = self.decoder4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(d3, e3)
        d3 = self.decoder3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.decoder2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.decoder1(torch.cat([d1, a1], dim=1))

        return self.final(d1)
    
    
def load_and_preprocess_image(image_path):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)
    
    # Apply CLAHE
#     clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(17,17))
#     image_np = clahe.apply(image_np)
    
    # Convert the NumPy array back to a PIL image
    image = Image.fromarray(image_np)
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    ])
    
    # Apply transformations
    tensor_image = transform(image)
    
    # Apply Gaussian Blur
    #tensor_image = F.gaussian_blur(tensor_image, kernel_size=(1, 1))  # Adjust kernel size as needed
    
    # Add batch dimension
    return tensor_image.unsqueeze(0)

def save_output(input_image, output_image, save_path):
    plt.figure(figsize=(12, 6))
    
    # Plot input image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title('Input Image (Grayscale 256x256)')
    plt.axis('off')
    
    # Plot output image
    plt.subplot(1, 2, 2)
    plt.imshow(output_image.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title('Generated Output')
    plt.axis('off')
    
    plt.show()  # Show the images in the notebook
    
    # Save only the output image to file
    output_array = output_image.squeeze().cpu().numpy()
    plt.imsave(save_path, output_array, cmap='gray')

def normalize_output(output):
    # Get min and max values from the output tensor for each image in the batch
    min_val = output.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Get minimum across the H and W dimensions
    max_val = output.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Get maximum across the H and W dimensions

    # Prevent division by zero in case all values are the same
    normalized_output = (output - min_val) / (max_val - min_val + 1e-8)  # Normalize to [0, 1]
    
    return normalized_output

def main():
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load('9_14_best_final.pth', map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Set up input and output directories
    #test_dir = 'Data/8_28_new/native'
    #test_dir = 'Data/8_30_new/native'
    #test_dir = 'Data/tests'
    #test_dir = 'Data/tests/9_2'
    #test_dir = 'Data/9_5/native'
    #test_dir='Data/9_11_tests/native'
    test_dir='Data/9_18_testb'
    output_dir = 'Data/outputs256'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the test directory
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Construct full file path
            image_path = os.path.join(test_dir, filename)
            
            # Load and preprocess the image
            input_image = load_and_preprocess_image(image_path).to(device)
            
            # Generate the output
            with torch.no_grad():
                output = model(input_image)
                
                # Normalize the output based on its min and max values
                output = normalize_output(output)
            
            # Save the output and display it
            save_path = os.path.join(output_dir, f"output_{filename.split('.')[0]}.png")
            save_output(input_image, output, save_path)
            print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    main()