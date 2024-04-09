import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import yaml
from imageClassifier import ImageClassifier
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, class_dict, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_dict = class_dict
        self.images = self._make_dataset()

    def _make_dataset(self):
        images = []
        for cls in self.class_dict.keys():
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                label = self.class_dict[cls]
                images.append((img_path, label))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

def load_class_dict_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        class_dict = yaml.safe_load(file)
    return class_dict

def train():
    class_dict = load_class_dict_from_yaml('class_mappings.yaml')
    
    height, width = 224, 224  # Assuming input image size
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    batch_size = 64
    
    train_loader = DataLoader(
        CustomDataset(root_dir='EuroSAT', class_dict=class_dict, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        CustomDataset(root_dir='data', class_dict=class_dict, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize the model
    num_classes = len(class_dict.keys())
    model = ImageClassifier(height, width, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    training_losses = []
    training_accuracies = []
    test_accuracies = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate training loss and accuracy
        epoch_loss = running_loss
        epoch_accuracy = correct / total
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)
        print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Evaluation phase (testing)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate test loss and accuracy
        test_loss = running_loss 
        test_accuracy = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f'Testing - Epoch [{epoch+1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
    
    print('Training finished!')
    save_path = "urban_model.pth"
    torch.save(model.state_dict(), save_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([i for i in range(len(training_losses))], training_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot([i for i in range(len(training_accuracies))], training_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.show()

if __name__ == "__main__":
    train()
