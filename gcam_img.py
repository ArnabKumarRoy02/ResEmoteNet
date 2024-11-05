import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np
from hook import Hook
from approach.ResEmoteNet import ResEmoteNet
from retinaface import RetinaFace

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using {device}")

class_labels = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Load the model
model = ResEmoteNet().to(device)
checkpoint = torch.load('best_model.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

final_layer = model.conv3
hook = Hook()
hook.register_hook(final_layer)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Text Settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (154, 1, 254)  # BGR color neon pink 254,1,154
thickness = 2
line_type = cv2.LINE_AA

transparency = 0.4

def detect_emotion(pil_crop_img):
    vid_fr_tensor = transform(pil_crop_img).unsqueeze(0).to(device)
    logits = model(vid_fr_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    predicted_class_idx = predicted_class.item()

    one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_()
    one_hot_output[0][predicted_class_idx] = 1
    logits.backward(one_hot_output, retain_graph=True)

    gradients = hook.backward_out
    feature_maps = hook.forward_out

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
    cam = cam.clamp(min=0).squeeze()

    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()

    scores = probabilities.cpu().detach().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores, cam

# Function to plot heatmap
def plot_heatmap(x, y, w, h, cam, pil_crop_img, image):
    # resize cam to w, h
    cam = cv2.resize(cam, (w, h))

    # apply color map to resized cam
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Get the region of interest on the image
    roi = image[y:y + h, x:x + w, :]

    # Blend the heatmap with the ROI
    overlay = heatmap * transparency + roi / 255 * (1 - transparency)
    overlay = np.clip(overlay, 0, 1)

    # Replace the ROI with the blended overlay
    image[y:y + h, x:x + w, :] = np.uint8(255 * overlay)

def update_max_emotion(rounded_scores):
    # Get index from max value in rounded_scores
    max_index = np.argmax(rounded_scores)
    max_emotion = class_labels[max_index]
    return max_emotion

def print_max_emotion(x, y, max_emotion, image):
    # Position to put the text for the max emotion
    org = (x, y - 15)
    cv2.putText(image, max_emotion, org, font, font_scale, font_color, thickness, line_type)

def print_all_emotion(x, y, w, rounded_scores, image):
    # Create text to be displayed
    org = (x + w + 10, y - 20)
    for index, value in enumerate(class_labels):
        emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
        y = org[1] + 40
        org = (org[0], y)
        cv2.putText(image, emotion_str, org, font, font_scale, font_color, thickness, line_type)

# Function to detect bounding box
def detect_bounding_box(image):
    faces = RetinaFace.detect_faces(image)
    for face_key in faces:
        face = faces[face_key]
        x1, y1, x2, y2 = face['facial_area']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        pil_crop_img = Image.fromarray(image[y1:y2, x1:x2])
        rounded_scores, cam = detect_emotion(pil_crop_img)

        max_emotion = update_max_emotion(rounded_scores)

        plot_heatmap(x1, y1, x2-x1, y2-y1, cam, pil_crop_img, image)
        print_max_emotion(x1, y1, max_emotion, image)
        print_all_emotion(x1, y1, x2-x1, rounded_scores, image)
    return faces

# Function to process a single image
def process_image(image_path):
    image = cv2.imread(image_path)
    faces = detect_bounding_box(image)
    cv2.imshow("ResEmoteNet Grad Cam", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hook.unregister_hook()

if __name__ == "__main__":
    image_path = "test_image.jpg"  
    process_image(image_path)
