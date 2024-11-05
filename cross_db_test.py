import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

from approach.ResEmoteNet import ResEmoteNet


# Set the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# Load the model
model = ResEmoteNet().to(device)
checkpoint = torch.load('best_model.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Apply the transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])


# Function to classify the image
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)
    scores = prob.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores


# Function to append the results to a list
def process_folder(folder_path):
    results = []
    for img_filename in os.listdir(folder_path):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_filename)
            scores = classify_image(img_path)
            results.append([img_path] + scores) 
    return results


def main(folder_path):
    results = process_folder(folder_path)
    header = ['filepath', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    with open('testfile.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
        

# Change the directory for the test folder        
main('path/to/test/folder')
