import os
import pandas as pd

path = '/path/to/train'

label_mapping = {
    "happy": 0,
    "surprise": 1,
    "sad": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5,
    "neutral": 6
}

image_data = []

# Make sure the name of the file is partition_iteration_emotion.jpg or .png
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        label_name = filename.split('_')[-1].split('.')[0]
        label_value = label_mapping.get(label_name)
        if label_value is not None:  
            image_data.append([filename, label_value])

df = pd.DataFrame(image_data, columns=["ImageName", "Label"])

csv_file_path = 'train_labels.csv'

df.to_csv(csv_file_path, index=False, header=False)

print(f"CSV file created at: {csv_file_path}")
