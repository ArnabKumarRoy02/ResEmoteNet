import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.fourforall import FourforAll
import cv2
from retinaface import RetinaFace

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Emotions labels
emotions = ['happiness', 'surprise', 'sadness', 'anger', 'fear', 'disgust']

model = FourforAll().to(device)
model.load_state_dict(torch.load('best_four4all.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Access the webcam
video_capture = cv2.VideoCapture(0)

# Settings for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 200, 0)  # This is BGR color
thickness = 3
line_type = cv2.LINE_AA

max_emotion = ''

def detect_emotion(video_frame):
    vid_fr_tensor = transform(video_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(vid_fr_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

def get_max_emotion(face, video_frame):
    x1, y1, x2, y2 = face['facial_area']
    crop_img = video_frame[y1:y2, x1:x2]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)    
    max_index = np.argmax(rounded_scores)
    max_emotion = emotions[max_index]
    return max_emotion

def print_max_emotion(face, video_frame, max_emotion):
    x1, y1, _, _ = face['facial_area']
    org = (x1, y1 - 15)
    cv2.putText(video_frame, max_emotion, org, font, font_scale, font_color, thickness, line_type)
    
def print_all_emotion(face, video_frame):
    x1, y1, x2, y2 = face['facial_area']
    crop_img = video_frame[y1:y2, x1:x2]
    pil_crop_img = Image.fromarray(crop_img)
    rounded_scores = detect_emotion(pil_crop_img)
    org = (x2 + 10, y1 - 20)
    for index, value in enumerate(emotions):
        emotion_str = (f'{value}: {rounded_scores[index]:.2f}')
        y = org[1] + 40
        org = (org[0], y)
        cv2.putText(video_frame, emotion_str, org, font, font_scale, font_color, thickness, line_type)

def detect_bounding_box(video_frame, counter):
    global max_emotion
    faces = RetinaFace.detect_faces(video_frame)
    for face_key in faces:
        face = faces[face_key]
        x1, y1, x2, y2 = face['facial_area']
        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if counter == 0:
            max_emotion = get_max_emotion(face, video_frame)
        print_max_emotion(face, video_frame, max_emotion)
        print_all_emotion(face, video_frame)
    return faces

def detect_stress_faces(video_frame, faces):
    for face_key in faces:
        face = faces[face_key]
        x1, y1, x2, y2 = face['facial_area']
        crop_img = video_frame[y1:y2, x1:x2]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = detect_emotion(pil_crop_img)
        max_index = np.argmax(rounded_scores)
        max_emotion = emotions[max_index]
        max_probability = rounded_scores[max_index]
        
        if max_emotion != 'happiness' and max_probability > 0.85:
            stress_label = 'Stressed'
            font_color = (0, 0, 255)
        elif max_emotion != 'happiness' and max_probability > 0.65:
            stress_label = 'Mildly Stressed'
            font_color = (0, 0, 175)
        else:
            stress_label = 'Not Stressed'
            font_color = (255, 0, 0)
            
        org = (x2 - 150, y1 - 15)  # Adjust the x-coordinate to be at the right side of the bounding box
        cv2.putText(video_frame, stress_label, org, font, font_scale, font_color, thickness, line_type)
    return video_frame

counter = 0
evaluation_frequency = 5

# Loop for Real-Time Face Detection
while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    
    if result is False:
        break  # Terminate the loop if the frame is not read successfully
    
    faces = detect_bounding_box(video_frame, counter)  # Apply the function we created to the video frame
    video_frame = detect_stress_faces(video_frame, faces)
    cv2.imshow("Four4All Stress Detection", video_frame)  # Display the processed frame in a window named "Four4All"
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    counter += 1
    
    if counter == evaluation_frequency:
        counter = 0
        
video_capture.release()
cv2.destroyAllWindows()