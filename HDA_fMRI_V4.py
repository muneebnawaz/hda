# Environment & core Imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import datasets, transforms

# Reproducibility
SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

# Batch Size
batch_size = 32

# Facial Images: Setting Directroy Path & Defining Trasformations
# Paths
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
                                       transforms.RandomHorizontalFlip(p=0.3),
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
fi_test_loader = DataLoader(fi_test_ds, batch_size = batch_size, shuffle = False)

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


# fMRI: Setting Paths

# Path Directories
fMRI_data_dir = "/home/muneeb/Data/rois_cc200"
fMRI_mapping_csv = "/home/muneeb/Data/1d_to_pheno_mapping.csv"


# Creating function to read .1D fMRI Timeseries
def LoadTimeseries(path):
  df = pd.read_csv(path, delim_whitespace=True, header=0)
  data = df.values
  return np.asarray(data, dtype=np.float32)

# Example usage of fMRI timeseries loading function
example_path = "/home/muneeb/Data/rois_cc200/NYU_0050960_rois_cc200.1D"
example_file = LoadTimeseries(example_path)
print("\nExample .1D file shape (Timepoints, ROIs):", example_file.shape)

'''
# plot first 5 ROIs
plt.figure(figsize=(10, 5))
for i in range(5):
  plt.plot(example_file[:, i], label=f"ROI {i+1}")
plt.legend()
#plt.show

# heatmap of all ROIs
plt.figure(figsize=(8, 8))
plt.imshow(example_file.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Intensity')
#plt.show()

# Correlation Matrix (Functional Connectivity)
plt.figure(figsize=(8, 8))
plt.imshow(np.corrcoef(example_file.T), cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
#plt.show()
'''

# Loading fMRI Phenotypic Data Mapping csv
pheno_csv = pd.read_csv(fMRI_mapping_csv)

# Loading all fMRI Timeseries & Labels
fMRI_all_data, fMRI_all_labels = [], []

for _, row in pheno_csv.iterrows():
  f_path = row["filepath"]
  label = 0 if row['DX_GROUP'] == 1 else 1

  f_ts_data = LoadTimeseries(f_path)
  fMRI_all_data.append(f_ts_data)
  fMRI_all_labels.append(label)

fMRI_all_data = np.asarray(fMRI_all_data, dtype=object)
fMRI_all_labels = np.asarray(fMRI_all_labels)

# Loaded Data check
print(f"\nLoaded fMRI {len(fMRI_all_data)} Timeseries")
print(f"Loaded fMRI {len(fMRI_all_labels)} Labels")
print(f"Example fMRI shape:", fMRI_all_data[0].shape)
print(f"fMRI Labels distribution -", np.unique(fMRI_all_labels, return_counts=True))

# Function to compute functional connectivity
def compute_fc_matrix(ts):
  fc = np.corrcoef(ts.T) # Transposed for ROI signals to move to rows
  fc = np.nan_to_num(fc, nan=0.0) # Replacing NaNs with zeros
  return fc.astype(np.float32)

# Convert all scans to FC matrices
all_data_fc = []
for ts_data in fMRI_all_data:
  fc_matrix = compute_fc_matrix(ts_data)
  all_data_fc.append(fc_matrix)
all_data_fc = np.array(all_data_fc, dtype=np.float32)
all_labels = np.array(fMRI_all_labels, dtype=np.int64)
print(f"\nAll FC matrices dataset shape:", all_data_fc.shape)

# Creating PyTorch Dataset for fMRI FC matrices and corresponding labels
class fMRIFCDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = torch.tensor(self.data[idx])
    y = torch.tensor(self.labels[idx])
    return x, y
fMRI_FC_Dataset = fMRIFCDataset(all_data_fc, all_labels)

# fMRI DataLoaders
'''
# Train/Test split
fMRI_train_size = int(0.85 * len(fMRI_FC_Dataset))
fMRI_test_size = len(fMRI_FC_Dataset) - fMRI_train_size
fMRI_train_dataset, fMRI_test_dataset = random_split(fMRI_FC_Dataset, [fMRI_train_size, fMRI_test_size], torch.Generator().manual_seed(SEED))

# DataLoaders
fMRI_train_loader = DataLoader(fMRI_train_dataset, batch_size = batch_size, shuffle = True, drop_last=True)
fMRI_test_loader = DataLoader(fMRI_test_dataset, batch_size = batch_size, shuffle = False)
print(f"fMRI Train samples:", len(fMRI_train_dataset))
print(f"fMRI Test samples:", len(fMRI_test_dataset))
'''
fMRI_loader = DataLoader(fMRI_FC_Dataset, batch_size = batch_size, shuffle = True, drop_last=True)

# Quick Check
batch_x, batch_y = next(iter(fMRI_loader))
print(f"\nfMRI Batch Shape:", {batch_x.shape})
print(f"fMRI Batch Labels:", {batch_y.shape})

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
fi_sample_input = torch.randn(32, 3, 224, 224)
fi_sample_output = face_embedding_model(fi_sample_input)
print("\nFacial Images Embedding Shape:", fi_sample_output.shape)

# fMRI FC Matrices Embedding Module
''' def normalize_adj(A):
    """
    A: (B, N, N) adjacency matrices
    Returns degree-normalized adjacency with self-loops
    """
    B, N, _ = A.size()
    I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
    A_hat = A + I  # add self-loops

    # Degree matrix (sum of rows)
    D = A_hat.sum(dim=2)  # shape (B, N)

    # D^(-1/2)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # shape (B, N, N)

    # Normalized adjacency: D^(-1/2) * A_hat * D^(-1/2)
    return torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt) '''


class FCMatrixEmbeddingModel(nn.Module):
    def __init__(self, num_nodes=200, hidden_dims=[256, 128], embedding_dim=128, dropout=0.5):
        super(FCMatrixEmbeddingModel, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        # Build multiple GCN-like layers
        self.gcn_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        input_dim = num_nodes
        for hdim in hidden_dims:
            self.gcn_layers.append(nn.Linear(input_dim, hdim))
            self.bns.append(nn.BatchNorm1d(hdim))
            input_dim = hdim

        # Final projection layer → embedding
        self.fc_out = nn.Linear(input_dim, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, adj):
        batch_size, N, _ = adj.size()

        # Initialize node features as identity (one-hot encoding of nodes)
        x = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Stacked "GCN" layers (without adjacency normalization)
        for layer, bn in zip(self.gcn_layers, self.bns):
            x = torch.bmm(adj, x)                      # propagate using raw adjacency
            x = layer(x)                               # linear transform
            x = bn(x.view(-1, x.size(-1))).view(batch_size, N, -1)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global average pooling → graph embedding
        x = x.mean(dim=1)

        # Final projection
        x = self.fc_out(x)
        x = F.relu(self.bn_out(x))
        return x

# Instantiate FC Matrix Embedding Model
fc_matrix_embedding_model = FCMatrixEmbeddingModel(hidden_dims=[512, 256, 128])

# Print Model Structure
#print(fc_matrix_embedding_model)

# Forward pass with dummy batch
#dummy_adj = torch.randn(32, 200, 200)
#dummy_output = fc_matrix_embedding_model(dummy_adj)
#print("FC Matrix Embedding Shape:", dummy_output.shape)

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
#print(classifier)

# Sanity Check
sample_embeddings = torch.randn(32, 128)
sample_logits = classifier(sample_embeddings)
print("Classifier Output shape:", sample_logits.shape)

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
params = list(face_embedding_model.parameters()) + list(fc_matrix_embedding_model.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Traning & Validation Loop

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

    # Set models to training mode
    fc_matrix_embedding_model.train()
    face_embedding_model.train()
    classifier.train()

    running_loss = 0.0
    running_cls_loss = 0.0
    running_mmd_loss = 0.0
    correct = 0
    total = 0

    # -----------------------------
    # Training Loop
    # -----------------------------
    for (fMRI_data, fMRI_labels), (fi_data, fi_labels) in zip(cycle(fMRI_loader), fi_train_loader):
        optimizer.zero_grad()

        # Forward pass: Embeddings
        fMRI_embeddings = fc_matrix_embedding_model(fMRI_data)
        fi_embeddings = face_embedding_model(fi_data)

        # Forward pass: Classifier
        fMRI_logits = classifier(fMRI_embeddings)
        fi_logits = classifier(fi_embeddings)

        # Classification Loss
        cls_loss = criterion(fMRI_logits, fMRI_labels) + criterion(fi_logits, fi_labels)

        # MMD Loss
        mmd_loss_val = mmd_loss(fMRI_embeddings, fi_embeddings)

        # Total Loss
        total_loss = cls_loss + lambda_mmd * mmd_loss_val

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Stats
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_mmd_loss += mmd_loss_val.item()
        total += fMRI_labels.size(0) + fi_labels.size(0)
        correct += (fMRI_logits.argmax(1) == fMRI_labels).sum().item() + \
                   (fi_logits.argmax(1) == fi_labels).sum().item()

    epoch_loss = running_loss / len(fMRI_loader)
    epoch_cls_loss = running_cls_loss / len(fMRI_loader)
    epoch_mmd_loss = running_mmd_loss / len(fMRI_loader)
    epoch_acc = correct / total

    # -----------------------------
    # Validation Loop
    # -----------------------------
    face_embedding_model.eval()
    classifier.eval()

    val_running_loss = 0.0
    val_running_cls_loss = 0.0
    val_running_mmd_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for fi_imgs, fi_labels in fi_val_loader:
            fi_embeddings = face_embedding_model(fi_imgs)
            fi_logits = classifier(fi_embeddings)

            cls_loss_fi = criterion(fi_logits, fi_labels)
            total_loss = cls_loss_fi  # No MMD for validation

            val_running_loss += total_loss.item()
            val_running_cls_loss += cls_loss_fi.item()
            val_total += fi_labels.size(0)
            val_correct += (fi_logits.argmax(1) == fi_labels).sum().item()

    val_epoch_loss = val_running_loss / len(fi_val_loader)
    val_epoch_cls  = val_running_cls_loss / len(fi_val_loader)
    val_epoch_acc  = val_correct / val_total

    print(f"\nEpoch [{epoch+1}/{num_epochs}]:")
    print(f"Train -> Total: {epoch_loss:.4f}, Cls: {epoch_cls_loss:.4f}, MMD: {epoch_mmd_loss:.4f}, Acc: {epoch_acc:.4f}")
    print(f"Val   -> Total: {val_epoch_loss:.4f}, Cls: {val_epoch_cls:.4f}, Acc: {val_epoch_acc:.4f}")

    # Step the scheduler based on validation accuracy
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
plt.savefig("V4_fMRI_confusion_matrix.png")
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
plt.savefig("V4_fMRI_roc_curve.png")
plt.close()