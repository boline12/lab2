import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset Selection & Preprocessing
# resize to 128x128, though CIFAR-10 is natively 32x32
# were tested and led to underfitting (failed augmentation). 

transform_train = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1] 
])

transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load full dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Split into 10% Labeled and 90% Unlabeled
num_train = len(full_train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)

labeled_size = int(0.1 * num_train)
labeled_indices = indices[:labeled_size]
unlabeled_indices = indices[labeled_size:]

labeled_dataset = Subset(full_train_dataset, labeled_indices)
unlabeled_dataset = Subset(full_train_dataset, unlabeled_indices)

labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Pipeline Design: Base Model
def get_model():
    # Use pre-trained ResNet18 and modify the final layer for 10 classes
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)

model = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Helper function to track metrics
history = {'stage1_loss': [], 'stage1_acc': [], 'stage2_loss': [], 'stage2_acc': []}

# 3. Model Training - Stage 1: Supervised
print("Starting Stage 1: Supervised Training on Labeled Data (10%)")
epochs_stage1 = 5

for epoch in range(epochs_stage1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in labeled_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(labeled_loader)
    epoch_acc = 100. * correct / total
    history['stage1_loss'].append(epoch_loss)
    history['stage1_acc'].append(epoch_acc)
    print(f"Epoch {epoch+1}/{epochs_stage1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

# 4. Pseudo-Labeling the Unlabeled Set
print("\nGenerating Pseudo-Labels...")
model.eval()
pseudo_images = []
pseudo_labels = []
confidence_threshold = 0.90 

with torch.no_grad():
    for inputs, _ in unlabeled_loader: # Ignoring actual labels
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, predicted = torch.max(probabilities, 1)
        
        mask = max_probs > confidence_threshold
        for i in range(len(mask)):
            if mask[i]:
                pseudo_images.append(inputs[i].cpu())
                pseudo_labels.append(predicted[i].cpu().item())

print(f"Generated {len(pseudo_labels)} confident pseudo-labels out of {len(unlabeled_indices)} unlabeled images.")

# Create a combined dataset
class PseudoDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

pseudo_dataset = PseudoDataset(pseudo_images, pseudo_labels)
combined_dataset = torch.utils.data.ConcatDataset([labeled_dataset, pseudo_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# 5. Model Training - Stage 2: Semi-Supervised
print("\nStarting Stage 2: Semi-Supervised Training (Labeled + Confident Pseudo-Labeled)")
epochs_stage2 = 5

for epoch in range(epochs_stage2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in combined_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(combined_loader)
    epoch_acc = 100. * correct / total
    history['stage2_loss'].append(epoch_loss)
    history['stage2_acc'].append(epoch_acc)
    print(f"Epoch {epoch+1}/{epochs_stage2} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

# 6. Evaluation and Visualization
print("\nEvaluating on Test Set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# Plotting Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['stage1_loss'] + history['stage2_loss'], label='Training Loss')
plt.axvline(x=epochs_stage1-1, color='r', linestyle='--', label='Pseudo-Labeling Added')
plt.title('Loss Curve across Stages')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['stage1_acc'] + history['stage2_acc'], label='Training Accuracy')
plt.axvline(x=epochs_stage1-1, color='r', linestyle='--', label='Pseudo-Labeling Added')
plt.title('Accuracy Curve across Stages')
plt.legend()
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.show()