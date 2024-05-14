from train import mnistNet
import torch
from PIL import Image
from torchvision import transforms

model = mnistNet()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

print("Inferencing...")
image_path = "7.png"

image = Image.open(image_path).convert('L')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image)
predicted_digit = torch.argmax(output, dim=1).item()
print(f"predicted_digit: {predicted_digit}")

