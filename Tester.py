import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import pandas as pd
from CustomDataset import CustomDataset
import cv2
import re
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Function to label test data (single folder, no class subfolders)
def datalabeling_single_folder(dataset_dir):
    image_data = []
    valid_extensions = ('.jpg', '.jpeg', '.png')  # Add valid image extensions
    for image_file in os.listdir(os.path.join(dataset_dir, 'FinalTest')):  # Use the 'FinalTest' folder
        if image_file.lower().endswith(valid_extensions):  # Only process valid image files
            # Add the 'FinalTest' folder as the class folder
            image_data.append([image_file, 'FinalTest', -1])  # Use 'FinalTest' as the folder
    df = pd.DataFrame(image_data, columns=['Image Name', 'Class Folder', 'Dummy'])
    return df


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test Dataset loading (all images in one folder)
testDataset_dir = './FinalTestDataset'
test_data_label = datalabeling_single_folder(testDataset_dir)

# Class names from training (assuming same class names from training)
class_names = ['중국_등광조망_이동', '중국_등광조망_조업', '중국_등광조망_표류', 
               '중국_범장망_이동', '중국_범장망_조업', '중국_범장망_표류', 
               '중국_유망_이동', '중국_유망_조업', '중국_유망_표류', 
               '중국_타망_이동', '중국_타망_조업', '중국_타망_표류', 
               '한국_낚시어선_이동', '한국_낚시어선_조업', '한국_낚시어선_표류', 
               '한국_안강망_이동', '한국_안강망_조업', '한국_안강망_표류', 
               '한국_연승_이동', '한국_연승_조업', '한국_연승_표류', 
               '한국_저인망_이동', '한국_저인망_조업', '한국_저인망_표류', 
               '한국_채낚이_이동', '한국_채낚이_조업', '한국_채낚이_표류', 
               '한국_통발_이동', '한국_통발_조업', '한국_통발_표류']

mean = [0.32546, 0.35472, 0.48292]
std = [0.21112, 0.20426, 0.24468]
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

testData = CustomDataset(test_data_label, testDataset_dir, transform=test_transform)
testLoader = DataLoader(testData, batch_size=4, shuffle=False)
print(f"Number of test samples: {len(testData)}")

# Load the model and adjust for the number of classes
model = models.densenet121(weights=None)
# model = models.resnet18(weights=None)
num_classes = len(class_names)  # Number of classes known from training
#model.fc = nn.Linear(model.fc.in_features, num_classes)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model.load_state_dict(torch.load('best_model_All_In.pth'))
model.to(device)

# Set the model to evaluation mode
model.eval()
criterion = nn.CrossEntropyLoss()

# Set the correct path to the NanumGothic font
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

# Longitude and latitude prefixes for searching
longitude_prefixes = ['123', '124', '125', '126', '127', '128', '129', '130', '138', '139']
latitude_prefixes = ['32', '34', '35', '36', '37', '38']

# OCR reader initialization
reader = easyocr.Reader(['en', 'ko'])

# Function for preprocessing the image for OCR
def preprocess_image(cropped_image):
    # 1. Resize
    scaled_image = cv2.resize(cropped_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # 2. Apply stronger sharpening filter
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9.3, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(scaled_image, -1, sharpen_kernel)

    # 3. Convert to grayscale
    gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

    # 4. Apply adaptive histogram equalization to improve contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)

    # 5. Apply denoising to remove noise
    denoised_image = cv2.fastNlMeansDenoising(equalized_image, None, 30, 7, 21)

    return denoised_image

# Function to format latitude/longitude
def format_lat_lon(value, prefix_length):
    prefix = value[:prefix_length]  # Take the first 'prefix_length' characters (e.g., 125 for LON)
    remainder = value[prefix_length:]  # The remaining part of the value

    # Remove '7', '8', '9' from the first 3 digits of the remainder
    first_three_digits = remainder[:3]
    valid_first_three_digits = re.sub(r'[789]', '', first_three_digits)

    valid_remainder = valid_first_three_digits + remainder[3:]
    
    if len(valid_remainder) > 2:
        formatted_value = f"{prefix}.{valid_remainder[:2]}.{valid_remainder[2:]}"
    else:
        formatted_value = f"{prefix}.{valid_remainder}"
    return formatted_value

# Function to find longitude and latitude in extracted text
def find_longitude_from_end(cleaned_text, longitude_prefixes):
    for prefix in longitude_prefixes:
        index = cleaned_text.rfind(prefix)
        if index != -1:
            lat_part = cleaned_text[:index]
            lon_part = cleaned_text[index:]
            if prefix in ['138', '139']:
                lon_part = '130' + lon_part[3:]
            return lat_part, lon_part
    return None, None


results = []

# Prediction and OCR process loop
with torch.no_grad():
    running_loss = 0.0
    total_images = 0
    correct_predictions = 0

    for i, (images, _) in enumerate(testLoader):
        images = images.to(device)
        print(f"Batch {i+1}: Processing {len(images)} images")

        # Forward pass: get predictions
        outputs = model(images)
        
        # Get predicted labels
        _, predicted = torch.max(outputs, 1)
        
        # Map predicted labels to class names
        predicted_class_names = [class_names[pred.item()] for pred in predicted]
        
        for j in range(len(predicted_class_names)):
            print(f"Predicted class name: {predicted_class_names[j]}")


            # Split the predicted class name into '선종' and '상태'
            split_class = predicted_class_names[j].split('_')
            type = split_class[1] if len(split_class) > 1 else ''
            status = split_class[2] if len(split_class) > 2 else ''

            # Get the image filename
            image_name = test_data_label.iloc[total_images + j, 0]

            # Perform OCR on the corresponding image
            img_path = os.path.join(testDataset_dir, 'FinalTest', test_data_label.iloc[total_images + j, 0])
            img = cv2.imread(img_path)

            if img is not None:
                height, width, _ = img.shape

                # Crop the bottom part of the image to focus on latitude/longitude text
                cropped_latlon = img[int(height * 0.95):height, int(width * 0.04):int(width * 0.4)]

                # Preprocess the cropped image for OCR
                processed_image = preprocess_image(cropped_latlon)

                # Display the preprocessed image (optional)
                # plt.imshow(processed_image, cmap='gray')
                # plt.title(f'Preprocessed Cropped Image - {img_path}', fontproperties=font_prop)
                # plt.show()

                # Perform OCR using EasyOCR
                extracted_text_easyocr = reader.readtext(processed_image, detail=0)
                joined_text = ' '.join(extracted_text_easyocr)

                # Clean the extracted text and find longitude and latitude
                cleaned_text = re.sub(r'[^0-9]', '', joined_text)
                lat_part, lon_part = find_longitude_from_end(cleaned_text, longitude_prefixes)

                if lat_part and lon_part:
                    formatted_lat = f"LAT N{format_lat_lon(lat_part, 2)}"
                    formatted_lon = f"LON E{format_lat_lon(lon_part, 3)}"
                    print(f"Formatted Coordinates: {formatted_lat} {formatted_lon}")
                else:
                    print("No valid longitude prefix found.")
            results.append([image_name, type, status, f'{formatted_lat} {formatted_lon}'])
        
        total_images += len(images)

df_results = pd.DataFrame(results,columns=['파일명','선종','상태','위치'])
df_results.to_csv('Predictions_AllIn_0.csv', index=False,encoding='utf-8-sig')
print("Prediction saved as 'Prediction_AllIn_0.csv'")
