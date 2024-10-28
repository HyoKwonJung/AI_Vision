import os
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Define the dataset directory
dataset_dir = 'fisheryDataset'  # Path to the main dataset folder containing class subfolders
train_dir = 'trainDataset'      # Path where you want to save training data
test_dir = 'testDataset'        # Path where you want to save testing data

# Ensure the output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the train-test split ratio
train_ratio = 0.85  # 85% training, 15% testing

# Loop over each class folder
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a directory
        # List all images in the current class
        images = os.listdir(class_path)

        if len(images) > 1:
            # Split images into train and test sets (only if there are 2 or more images)
            train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
        else:
            # If there's only 1 image, assign it to the train set by default
            train_images = images
            test_images = []


        # Create class-specific directories in train and test folders
        train_class_dir = os.path.join(train_dir, class_folder)
        test_class_dir = os.path.join(test_dir, class_folder)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copy training images to the train directory
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            copyfile(src, dst)

        # Copy testing images to the test directory
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_class_dir, img)
            copyfile(src, dst)

        print(f"Class '{class_folder}' - Train: {len(train_images)} | Test: {len(test_images)}")
