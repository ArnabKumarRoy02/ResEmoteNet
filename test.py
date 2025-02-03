import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from approach.ResEmoteNet import ResEmoteNet
from get_dataset import Four4All

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Transform the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Testing Data
fer_dataset_test = Four4All(csv_file='/path/to/test_labels.csv',
                            img_dir='/path/to/test/data', transform=transform)

data_test_loader = DataLoader(fer_dataset_test, batch_size=16, shuffle=False)
test_image, test_label = next(iter(data_test_loader))

# Criterion
criterion = torch.nn.CrossEntropyLoss()

# Load the best model before testing
model = ResEmoteNet()
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Final test with the best model
model.eval()
final_test_loss = 0.0
final_test_correct = 0
final_test_total = 0
with torch.no_grad():
    for data in tqdm(data_test_loader, desc="Final Testing with Best Model"):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        final_test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        final_test_total += labels.size(0)
        final_test_correct += (predicted == labels).sum().item()

final_test_loss = final_test_loss / len(data_test_loader)
final_test_acc = final_test_correct / final_test_total

print(f"Final Test Loss: {final_test_loss}, \nFinal Test Accuracy: {final_test_acc}")