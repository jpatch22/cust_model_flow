import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

# Import the model from imageClassifier.py
from imageClassifier import ImageClassifier

# Define parameters
height = 32
width = 32
num_classes = 10
batch_size = 32
epochs = 1  # Set to 1 for now, but can be flexible

print("creating model")
model = ImageClassifier(height, width, num_classes)
print("model created")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(100):  # Assuming 100 batches
        # Generate random input data and labels
        inputs = torch.randn(batch_size, 3, height, width)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

# Save the trained model
save_path = 'image_classifier.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved at: {save_path}")

