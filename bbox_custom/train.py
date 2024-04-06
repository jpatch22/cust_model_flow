import torch
import torch.nn as nn
import torch.optim as optim
import airplaneDetector

# Generate random labels for classification and bounding box regression
def generate_random_labels(batch_size):
    cls_labels = torch.randint(0, 2, (batch_size,))
    bb_labels = torch.randn(batch_size, 4)  # Assuming 4 parameters for bounding box (x, y, width, height)
    return cls_labels, bb_labels

# Define parameters
num_epochs = 1
batch_size = 1
height = 128
width = 128
num_channels = 3
learning_rate = 0.001

# Create model instance
model = airplaneDetector.AirplaneDetector(width=width, height=height)

# Define loss function and optimizer
criterion_cls = nn.CrossEntropyLoss()
criterion_bb = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear gradients

    input_data = torch.randn(batch_size, num_channels, width, height)  # Random data
    cls_labels, bb_labels = generate_random_labels(batch_size)  # Random labels

    # Forward pass
    cls_outputs, bb_outputs = model(input_data)
    print(cls_outputs.shape, bb_outputs.shape)

    # Check shapes
    print(cls_outputs.shape[0])
    assert cls_outputs.shape[0] == batch_size, "Output size mismatch: Classification"
    assert bb_outputs.shape[0] == batch_size, "Output size mismatch: Bounding Box Regression"

    # Compute losses
    cls_loss = criterion_cls(cls_outputs, cls_labels)
    bb_loss = criterion_bb(bb_outputs, bb_labels)

    # Backpropagation
    total_loss = cls_loss + bb_loss
    total_loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved as trained_model.pth")

