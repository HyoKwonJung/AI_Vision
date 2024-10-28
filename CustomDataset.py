import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image  

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pd_file, root_dir, transform=None):
        self.annotation = pd_file  # This is a Pandas DataFrame
        self.root_dir = root_dir   # The root directory where the class folders are located
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        # The first column contains the image file, the second contains the class folder, the third contains the class label
        class_folder = self.annotation.iloc[index, 1]
        image_file = self.annotation.iloc[index, 0]
        class_label = self.annotation.iloc[index, 2]  # Use the class label as an integer
        
        # Construct the full image path
        image_path = os.path.join(self.root_dir, str(class_folder), image_file)
        
        # Print image path for debugging
        # print(f"Loading image from: {image_path}")
        
        # Load the image
        image = cv2.imread(image_path)
        
        # If the image couldn't be loaded, raise an error
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to PIL format (if you're using torchvision transforms)
        image = Image.fromarray(image)

        # Convert class label to a PyTorch tensor (it's already an integer)
        y_label = torch.tensor(class_label, dtype=torch.long)

        # Apply any transformations (if provided)
        if self.transform:
            image = self.transform(image)

        return image, y_label

