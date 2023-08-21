import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Define a simple classifier using a pre-trained ResNet-18 backbone
class MNISTResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(MNISTResNet18Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)


def test_model():
     for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        return loss


# Define the transformation for data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to match the input size of ResNet-18
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels (RGB)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading all data into dataset
train_dataset = torchvision.datasets.ImageFolder(root='melanoma_cancer_dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root='melanoma_cancer_dataset/test', transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the model and define loss function and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device:", device)
model = MNISTResNet18Classifier(num_classes=2).to(device)  # 10 classes for MNIST
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
iterations = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iterations += 1
    if iterations % 10: 
      print(iterations, 'Loss:',loss.item())
    if iterations % 100: 
        test_model(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), 'mnist_resnet18_classifier.pth')

# Dataset Link
# https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
