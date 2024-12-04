import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from CustomDataset import CustomDataset

def datalabeling(dataset_dir):
    image_data = []
    for class_label, class_folder in enumerate(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_folder)
        
        if os.path.isdir(class_path):
            # List all images in the current class
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

                # Ensure it's a file (ignore directories or hidden files)
                if os.path.isfile(image_path):
                    # Save both the image file name, the class folder, and the class label (integer)
                    image_data.append([image_file, class_folder, class_label])

    # Return the DataFrame with image names, class folders, and class labels
    df = pd.DataFrame(image_data, columns=['Image Name', 'Class Folder', 'Class Label'])
    return df

trainDataset_dir = './trainDataset'

training_data_label = datalabeling(trainDataset_dir)
# print(training_data_label)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor()  # Convert the image to a PyTorch Tensor
])


trainingData = CustomDataset(training_data_label,trainDataset_dir,transform = transform)

imgs = [item[0] for item in trainingData]
imgs = torch.stack(imgs,dim=0).numpy()
# print(imgs)

mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()

std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()

print(f"Mean RGB : {mean_r}, {mean_g}, {mean_b}")
print(f"Std RGB : {std_r}, {std_g}, {std_b}")


# Mean RGB : 0.32546156644821167, 0.3547220230102539, 0.48292574286460876
# Std RGB : 0.21112343668937683, 0.20426224172115326, 0.24468673765659332