# Environment & core Imports
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
from torchvision import datasets, transforms

#Reproducibility
SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

# Batch Size
batch_size = 32

# Facial Images: Setting Directroy Path & Defining Trasformations
# Path
fi_train_dir = "/home/muneeb/Data/FI_new_Split/train"
fi_val_dir = "/home/muneeb/Data/FI_new_Split/val"
fi_test_dir = "/home/muneeb/Data/FI_new_Split/test"

# Parameters
fi_img_size = (224, 224) # Resize to match MobileNet input

# Define Transformations
fi_val_test_transform = transforms.Compose([transforms.Resize(fi_img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean = [0.485, 0.456, 0.408],
                                                       std = [0.229, 0.224, 0.225])
                                                       ])

fi_train_transform = transforms.Compose([transforms.Resize(fi_img_size),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation(degrees=10),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.408],
                                                           std=[0.229, 0.224, 0.225])
                                                           ])

# Augmentation if required
# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)


# Loading Facial Images Dataset
fi_train_ds = datasets.ImageFolder(root=fi_train_dir, transform=fi_train_transform)
fi_val_ds = datasets.ImageFolder(root=fi_val_dir, transform=fi_val_test_transform)
fi_test_ds = datasets.ImageFolder(root=fi_test_dir, transform=fi_val_test_transform)

# Facial images DataLoaders
fi_train_loader = DataLoader(fi_train_ds, batch_size = batch_size, shuffle = True, drop_last=True)
fi_val_loader = DataLoader(fi_val_ds, batch_size = batch_size, shuffle = False)
fi_test_loader = DataLoader(fi_test_ds, batch_size = batch_size, shuffle = False,)

# Sanity checks

# Dataset Sizes
print(f"\nFacial Images Train: {len(fi_train_ds)} Images")
print(f"Facial Images Validation: {len(fi_val_ds)} Images")
print(f"Facial Images Test: {len(fi_test_ds)} Images")

# Facial Image DataLoader Batches
print("\nFacial Image DataLoader Batches:")
print(f"Train: {len(fi_train_loader)} batches")
print(f"Validation: {len(fi_val_loader)} batches")
print(f"Test: {len(fi_test_loader)} batches")

# Batch shape check
xb, yb = next(iter(fi_train_loader))
print(f"\nFacial Images Batch Shape: {xb.shape}")
print(f"Facial Images Batch Labels: {yb.shape}")

# EEG: Setting Directroy Paths & Defining Trasformations

# Path & Parameters
eeg_train_dir = "/home/muneeb/Data/EEG_data"
#eeg_test_dir = "/home/muneeb/Data/eeg_dataset/test"
eeg_img_size = (224, 224)
#eeg_val_split = 0.1

# Define Transformations
eeg_tfms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.Resize(eeg_img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.7703,), std=(0.0835,))])

# Loading EEG Spectograms into Datasets
eeg_train_full = datasets.ImageFolder(eeg_train_dir, transform = eeg_tfms)

'''
# Create validation split
n_total = len(eeg_train_full)
n_val   = int(n_total * eeg_val_split)
n_train = n_total - n_val
eeg_train_ds, eeg_val_ds = random_split(eeg_train_full, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))
                                '''

#eeg_test_ds = datasets.ImageFolder(eeg_test_dir, transform = eeg_tfms)

# EEG DataLoaders

eeg_train_loader = DataLoader(eeg_train_full, batch_size = batch_size, shuffle = True, drop_last=True)
#eeg_val_loader = DataLoader(eeg_val_ds, batch_size = batch_size, shuffle = False, drop_last=True)
#eeg_test_loader = DataLoader(eeg_test_ds, batch_size = batch_size, shuffle = False, drop_last=True)

# Sanity checks

# Dataset Sizes
print(f"\nEEG Train: {len(eeg_train_full)} Images")
#print(f"EEG Validation: {len(eeg_val_ds)} Images")
#print(f"EEG Test: {len(eeg_test_ds)} Images")

# EEG DataLoader Batches
print("\nEEG DataLoader Batches:")
print(f"Train: {len(eeg_train_loader)} batches")
#print(f"Validation: {len(eeg_val_loader)} batches")
#print(f"Test: {len(eeg_test_loader)} batches")

# Batch shape check
eeg_xb, eeg_yb = next(iter(eeg_train_loader))
print(f"\nEEG Batch Shape: {eeg_xb.shape}")
print(f"EEG Batch Labels: {eeg_yb.shape}")

# Facial Images Embedding Module

# Load pretrained ConvNeXt-Tiny
convnext_model = models.convnext_tiny(weights="IMAGENET1K_V1")

# Embedding Module
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()

        # Use the feature extractor part (drop the classifier)
        self.features = nn.Sequential(*list(convnext_model.children())[:-1])  # [B, 768, 7, 7]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, embedding_dim)  # ConvNeXt-Tiny outputs 768 channels
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)       # [B, 768, 7, 7]
        x = self.pool(x)           # [B, 768, 1, 1]
        x = torch.flatten(x, 1)    # [B, 768]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

# Instantiate Face Embedding Model
face_embedding_model = FaceEmbeddingModel()

# Sanity check through dummy input
fi_sample_input = torch.randn(2, 3, 224, 224)
fi_sample_output = face_embedding_model(fi_sample_input)
print("\nFacial Images Embedding Shape:", fi_sample_output.shape)

# EEG Embedding Module

class EEGEmbeddingModel(nn.Module):
  def __init__(self, embedding_dim=128):
    super(EEGEmbeddingModel, self).__init__()

    self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    self.pool2 = nn.AdaptiveAvgPool2d((4,4))
    
    self.fc1 = nn.Linear(256 * 4 * 4, 512)
    self.bn_fc1 = nn.BatchNorm1d(512)
    self.dropout = nn.Dropout(0.4)

    self.fc2 = nn.Linear(512, embedding_dim)
    self.bn_fc2 = nn.BatchNorm1d(embedding_dim)

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)

    x = self.pool2(x)
    x = torch.flatten(x, 1)

    x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
    x = F.relu(self.bn_fc2(self.fc2(x)))
    return x

# Instantiate EEG Embedding Model
eeg_embedding_model = EEGEmbeddingModel()

# Sanity check through dummy input
eeg_sample_input = torch.randn(2, 1, 224, 224)
eeg_sample_output = eeg_embedding_model(eeg_sample_input)
print("\nEEG Embedding Shape:", eeg_sample_output.shape)

# Classifier Head

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2, dropout=0.4):
        super(EmbeddingClassifier, self).__init__()

        # Fully connected VGG-style blocks
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Output layer
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc_out(x)
        return x

# Instantiate Classifier

classifier = EmbeddingClassifier()

# Sanity Check

sample_embeddings = torch.randn(4, 128)
sample_logits = classifier(sample_embeddings)
print("\nOutput shape:", sample_logits.shape)

# Gaussian Kernel Calculator

def gaussian_kernel(x,y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

  #Returns a gaussian kernel

  n_samples = x.size(0) + y.size(0)
  total = torch.cat([x,y], dim=0)

  L2_distance = ((total.unsqueeze(1) - total.unsqueeze(0))**2).sum(2)

  if fix_sigma:
    bandwidth = fix_sigma
  else:
    bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

  bandwidth /= kernel_mul ** (kernel_num // 2)
  bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

  kernel_val = [torch.exp(-L2_distance / b) for b in bandwidth_list]
  return sum(kernel_val) / len(kernel_val)

# MMD Loss Function

def mmd_loss(source, target):

  batch_size = source.size(0)

  kernels = gaussian_kernel(source, target)

  XX = kernels[:batch_size, :batch_size]
  YY = kernels[batch_size:, batch_size:]
  XY = kernels[:batch_size, batch_size:]
  YX = kernels[batch_size:, :batch_size]

  return torch.mean(XX + YY - XY - YX)

# Training Setup

num_epochs = 20
learning_rate = 0.001
lambda_mmd = 0.1

# Epochs for frozen backbone phase
freeze_epochs = 5

# Optimizer
params = list(eeg_embedding_model.parameters()) + list(face_embedding_model.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Training & validation Loop
for epoch in range(num_epochs):

    # -----------------------------
    # Phase 1: Freeze ConvNeXt backbone (first few epochs)
    # Phase 2: Unfreeze for fine-tuning (after freeze_epochs)
    # -----------------------------
    if epoch < freeze_epochs:
        for param in face_embedding_model.features.parameters():
            param.requires_grad = False
        if epoch == 0:
            print("Phase 1: Frozen ConvNeXt backbone for initial training.")
    else:
        for param in face_embedding_model.features.parameters():
            param.requires_grad = True
        if epoch == freeze_epochs:
            print("Phase 2: Unfreezing ConvNeXt backbone for fine-tuning.")

    
    # Training
    eeg_embedding_model.train()
    face_embedding_model.train()
    classifier.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_mmd_loss = 0.0
    correct = 0
    total = 0

    for (eeg_imgs, eeg_labels), (fi_imgs, fi_labels) in zip(cycle(eeg_train_loader), fi_train_loader):
        
        optimizer.zero_grad()

        # forward pass: Embeddings
        eeg_embed = eeg_embedding_model(eeg_imgs)
        fi_embed = face_embedding_model(fi_imgs)

        # forward pass: Classifier
        eeg_logits = classifier(eeg_embed)
        fi_logits = classifier(fi_embed)

        # Classification Loss
        cls_loss_eeg = criterion(eeg_logits, eeg_labels)
        cls_loss_fi = criterion(fi_logits, fi_labels)

        # MMD Loss
        mmd_loss_val = mmd_loss(eeg_embed, fi_embed)

        # Total Loss
        total_loss = cls_loss_eeg + cls_loss_fi + lambda_mmd * mmd_loss_val

        # Backward & Optimize
        total_loss.backward()
        optimizer.step()

        # Stats
        running_loss += total_loss.item()
        running_cls_loss += cls_loss_eeg.item() + cls_loss_fi.item()
        running_mmd_loss += mmd_loss_val.item()
        total += eeg_labels.size(0) + fi_labels.size(0)
        correct += (eeg_logits.argmax(1) == eeg_labels).sum().item() + (fi_logits.argmax(1) == fi_labels).sum().item()

    epoch_loss = running_loss / len(fi_train_loader)
    epoch_cls  = running_cls_loss / len(fi_train_loader)
    epoch_mmd  = running_mmd_loss / len(fi_train_loader)
    epoch_acc  = correct / total

    # Validation
    #eeg_embedding_model.eval()
    face_embedding_model.eval()
    classifier.eval()

    val_running_loss = 0.0
    val_running_cls_loss = 0.0
    val_running_mmd_loss = 0.0
    val_correct = 0
    val_total = 0
    '''
    with torch.no_grad():
        for (eeg_imgs, eeg_labels), (fi_imgs, fi_labels) in zip(cycle(eeg_val_loader), fi_val_loader):
            eeg_embed = eeg_embedding_model(eeg_imgs)
            fi_embed = face_embedding_model(fi_imgs)

            eeg_logits = classifier(eeg_embed)
            fi_logits = classifier(fi_embed)

            cls_loss_eeg = criterion(eeg_logits, eeg_labels)
            cls_loss_fi = criterion(fi_logits, fi_labels)
            mmd_loss_val = mmd_loss(eeg_embed, fi_embed)

            total_loss = cls_loss_eeg + cls_loss_fi + lambda_mmd * mmd_loss_val
            
    val_epoch_loss = val_running_loss / len(fi_val_loader)
    val_epoch_cls  = val_running_cls_loss / len(fi_val_loader)
    val_epoch_mmd  = val_running_mmd_loss / len(fi_val_loader)
    val_epoch_acc  = val_correct / val_total
    '''

    with torch.no_grad():
        for fi_imgs, fi_labels in fi_val_loader:
            fi_embeddings = face_embedding_model(fi_imgs)
            fi_logits = classifier(fi_embeddings)

            cls_loss_fi = criterion(fi_logits, fi_labels)
            total_loss = cls_loss_fi

            # Stats
            val_running_loss += total_loss.item()
            val_running_cls_loss += cls_loss_fi.item()
            val_total += fi_labels.size(0)
            val_correct += (fi_logits.argmax(1) == fi_labels).sum().item()

    val_epoch_loss = val_running_loss / len(fi_val_loader)
    val_epoch_cls  = val_running_cls_loss / len(fi_val_loader)
    val_epoch_acc  = val_correct / val_total


    print(f"\nEpoch [{epoch+1}/{num_epochs}]:")
    print(f"Train -> Total: {epoch_loss:.4f}, Cls: {epoch_cls:.4f}, MMD: {epoch_mmd:.4f}, Acc: {epoch_acc:.4f}")
    print(f"Val   -> Total: {val_epoch_loss:.4f}, Cls: {val_epoch_cls:.4f}, Acc: {val_epoch_acc:.4f}")

    # Scheduler based on Validation accuracy
    scheduler.step(val_epoch_acc)



# HDA TESTING
face_embedding_model.eval()
classifier.eval()

fi_correct = 0
fi_total = 0
fi_y_true = []
fi_y_pred = []
fi_y_prob = []

with torch.no_grad():
       
    for fi_imgs, fi_labels in fi_test_loader:

        fi_features = face_embedding_model(fi_imgs)
        fi_outputs = classifier(fi_features)
        fi_probs = F.softmax(fi_outputs, dim=1)[:,1]

        _, fi_predicted = torch.max(fi_outputs.data, 1)

        fi_y_true.extend(fi_labels.numpy())
        fi_y_pred.extend(fi_predicted.numpy())
        fi_y_prob.extend(fi_probs.numpy())


        fi_total += fi_labels.size(0)
        fi_correct += (fi_predicted == fi_labels).sum().item()


# Accuracy
fi_accuracy = 100 * fi_correct / fi_total


# Confusion Matrix and Metrics
fi_y_true = np.array(fi_y_true)
fi_y_pred = np.array(fi_y_pred)
fi_y_prob = np.array(fi_y_prob)

TP = np.sum((fi_y_true == 1) & (fi_y_pred == 1))
TN = np.sum((fi_y_true == 0) & (fi_y_pred == 0))
FP = np.sum((fi_y_true == 0) & (fi_y_pred == 1))
FN = np.sum((fi_y_true == 1) & (fi_y_pred == 0))

precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
fpr_value = FP / (FP + TN + 1e-8)

# ROC Curve & AUC
# Sort by predicted probabilities
sorted_indices = np.argsort(-fi_y_prob)
y_true_sorted = fi_y_true[sorted_indices]
y_prob_sorted = fi_y_prob[sorted_indices]

tpr_list = []
fpr_list = []
TP_cum, FP_cum = 0, 0
P = np.sum(fi_y_true)
N = len(fi_y_true) - P

for i in range(len(y_prob_sorted)):
    if y_true_sorted[i] == 1:
        TP_cum += 1
    else:
        FP_cum += 1
    tpr_list.append(TP_cum / P if P else 0)
    fpr_list.append(FP_cum / N if N else 0)

tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)
auc_value = np.trapezoid(tpr_list, fpr_list)

# Print Metrics 
print("\nFacial Image Test Results")
print(f"Accuracy : {fi_accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"FPR      : {fpr_value:.4f}")
print(f"AUC      : {auc_value:.4f}")

# Confusion Matrix Plot
cm = np.array([[TN, FP], [FN, TP]])
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Facial Images)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("confusion_matrix_EEG_V5.png")
plt.close()

# ROC Curve Plot
plt.figure(figsize=(6, 5))
plt.plot(fpr_list, tpr_list, label=f"AUC = {auc_value:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Facial Images)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_EEG_V5.png")
plt.close()