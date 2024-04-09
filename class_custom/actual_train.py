import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from imageClassifier import ImageClassifier
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._make_dataset()

    def _make_dataset(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

def train():
    height, width = 224, 224  # Assuming input image size
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and DataLoader
    dataset = CustomDataset(root_dir='data', transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Initialize the model
    num_classes = len(dataset.classes)
    model = ImageClassifier(height, width, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
   
    training_losses = []
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
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataset)
        training_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    print('Training finished!')
    save_path = "urban_model.pth"
    torch.save(model.state_dict(), save_path)
    plt.plot([i for i in range(len(training_losses))], training_losses)
    plt.show()

def main():
    height, width = 224, 224  # Assuming input image size
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomDataset(root_dir='data', transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    for img, label in dataset:
        print(label)

if __name__ == "__main__":
    train()
