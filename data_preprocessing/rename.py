import os
import shutil

# Define the source directory
source_dir = 'data_dir'

# Define the destination directory
destination_dir = 'out_dir'

# Mapping for folder names
folder_mapping = {'test': 'test', 
                  'train': 'train', 
                  'validation': 'val'}

# Iterate over the folders (test, train, validation)
for folder in ['test', 'train', 'validation']:
    folder_path = os.path.join(source_dir, folder)
    
    # Iterate over the class folders inside each folder
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        
        # Iterate over the images in the class folder
        for index, image in enumerate(os.listdir(class_folder_path)):
            # Get the image name and extension
            image_name, image_ext = os.path.splitext(image)
            
            # Construct the new image name
            new_image_name = f"{folder_mapping[folder]}_{index}_{class_folder}{image_ext}"
            
            # Rename the image and move it to the destination folder
            shutil.move(os.path.join(class_folder_path, image), os.path.join(destination_dir, folder, new_image_name))