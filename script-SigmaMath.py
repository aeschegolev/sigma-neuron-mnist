import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HIDEM=500

# Libraries for plotting graphs and calculating scientific metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# =====================================================================
# 1. DEVICE SETUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in Use: {device}")

# =====================================================================
# 2. DATA PREPARATION
# =====================================================================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# =====================================================================
# 3. ARCHITECTURE (MNIST(28*28) + H=500 + DynamicRangeBorders + BATCHNORM AND DROPOUT)
# =====================================================================
class FinalPerfectPerceptron(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=HIDEM, output_dim=10):
        super(FinalPerfectPerceptron, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(0.2) 
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU() 
        
        self.w_min = -1.0 / HIDEM
        self.w_max = 1.0 / HIDEM
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, self.w_min, self.w_max)
            nn.init.uniform_(self.fc2.weight, self.w_min, self.w_max)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.relu(out) 

    def apply_weight_clipping(self):
        with torch.no_grad():
            self.fc1.weight.clamp_(self.w_min, self.w_max)
            self.fc2.weight.clamp_(self.w_min, self.w_max)

model = FinalPerfectPerceptron().to(device)

# =====================================================================
# 4. LOSS FUNCTION, OPTIMIZER, AND LR PLANNER
# =====================================================================
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.005) 

epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

# Lists for logging metrics (needed for the article's charts)
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

# =====================================================================
# 5. TRAINING AND VALIDATION CYCLE
# =====================================================================
print("Start of Training...")

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    epoch_train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
        
        outputs = model(images)
        loss = criterion(outputs, one_hot_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.apply_weight_clipping()
        
        epoch_train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    scheduler.step()
    
    # --- INTERIM TESTING PHASE ---
    model.eval()
    epoch_test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            one_hot_labels = nn.functional.one_hot(labels, num_classes=10).float()
            
            outputs = model(images)
            loss = criterion(outputs, one_hot_labels)
            
            epoch_test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Saving Metrics to History
    t_loss = epoch_train_loss / len(train_loader.dataset)
    t_acc = (train_correct / train_total) * 100
    v_loss = epoch_test_loss / len(test_loader.dataset)
    v_acc = (test_correct / test_total) * 100
    
    history['train_loss'].append(t_loss)
    history['train_acc'].append(t_acc)
    history['test_loss'].append(v_loss)
    history['test_acc'].append(v_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d}/{epochs:02d} | LR: {current_lr:.5f} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | Test Acc: {v_acc:.2f}%")

# =====================================================================
# 6. GENERATION OF ACADEMIC METRICS
# =====================================================================
print("\n" + "="*50)
print(" GENERATING FINAL METRICS ")
print("="*50)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# 1. Text Report: Precision, Recall, and F1-Score for Each Class (Digit)
print("\n📋 Classification Report:")
print(classification_report(all_targets, all_preds, digits=4, target_names=[str(i) for i in range(10)]))

# 2. Confusion Matrix Output
print("📊 Confusion Matrix:")
cm = confusion_matrix(all_targets, all_preds)
print(cm)

# 3. CREATING GRAPHS (Saved to files in the project folder)
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(14, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, history['test_loss'], label='Test Loss', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Loss Curves', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('MSE Loss', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', color='#2ca02c', linewidth=2)
plt.plot(epochs_range, history['test_acc'], label='Test Accuracy', color='#d62728', linestyle='--', linewidth=2)
plt.title('Accuracy Curve', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300) # Saving in high quality for the article
print("\n💾 The graph of the learning curves is saved as 'learning_curves.png'")

# Visualizing and Saving the Error Matrix
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.savefig('confusion_matrix.png', dpi=300)
print("💾 The confusion matrix graph is saved as 'confusion_matrix.png'\n")
