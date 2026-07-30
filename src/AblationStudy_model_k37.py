#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import json
import glob
import shutil
import os


# In[1]:


# Reading data

path = '/work/bb1070/b382483/Final/L=2km_new/AblationStudy/model_k37'

# Create a directory to save the frames
os.makedirs(path, exist_ok=True)


# In[3]:


# Define the paths
path1 = "/work/bb1070/b382483/Data/TrainTest/ftp_stability/features_*"
path2 = "/work/bb1070/b382483/Data/TrainTest/ftp_stability/labels_*"

# Use glob to get all the files matching the pattern
files1 = sorted(glob.glob(path1))
files2 = sorted(glob.glob(path2))


# In[4]:


# Load a sample feature and label file and print their shapes
sample_feature = xr.open_dataset(files1[0])
sample_label = xr.open_dataset(files2[0])
print("Shape of features before creating Dataset:", sample_feature.isel(Time=0).to_array().values.shape)
print("Shape of labels before creating Dataset:", sample_label.isel(Time=0).to_array().values.shape)


# In[5]:


print(sample_label)


# In[6]:


sample_feature = xr.open_dataset(files1[0])
sample_label = xr.open_dataset(files2[0])
print("Shape of features before creating Dataset:", sample_feature.to_array().values.shape)
print("Shape of labels before creating Dataset:", sample_label.to_array().values.shape)


# In[7]:


time_steps_per_file = len(sample_feature['Time'])

# Create indices for all samples
num_files = len(files1)
total_samples = num_files * time_steps_per_file
indices = np.arange(total_samples)


# In[8]:


# Shuffle and split the indices
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2


# In[9]:


# Print statements for debugging
print("Number of time steps per file:", time_steps_per_file)
print("Length of features files:", len(files1))
print("Length of labels files:", len(files2))
print("Total number of samples:", total_samples)
print("Train indices:", len(train_indices))
print("Validation indices:", len(val_indices))
print("Test indices:", len(test_indices))


# In[10]:


def get_file_and_time_step(index):
    file_index = index // time_steps_per_file
    time_step_index = index % time_steps_per_file
    return file_index, time_step_index


# # Loading data 

# In[11]:


class CustomDataset(Dataset):
    def __init__(self, indices, feature_files, label_files):
        self.indices = indices
        self.feature_files = feature_files
        self.label_files = label_files

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        overall_index = self.indices[idx]
        file_index, time_step_index = get_file_and_time_step(overall_index)

        with xr.open_dataset(self.feature_files[file_index]) as ds_feature:
            X = ds_feature.isel(Time=time_step_index).to_array().values
        with xr.open_dataset(self.label_files[file_index]) as ds_label:
            y = ds_label.isel(Time=time_step_index).to_array().values

        # Check if there's an extra dimension and handle it
        if X.ndim == 4 and X.shape[0] == 1:
            X = X.squeeze(0)

        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)


        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor


# In[12]:


# Create Dataset objects
train_dataset = CustomDataset(train_indices, files1, files2)
val_dataset = CustomDataset(val_indices, files1, files2)
test_dataset = CustomDataset(test_indices, files1, files2)


# In[13]:


# Create DataLoader objects
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


# # CNN model

# In[ ]:


class ConcatConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_k3=True, use_k5=True, use_k7=True):
        super().__init__()

        self.use_k3 = use_k3
        self.use_k5 = use_k5
        self.use_k7 = use_k7

        if use_k3:
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if use_k5:
            self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        if use_k7:
            self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []

        if self.use_k3:
            outputs.append(torch.relu(self.conv3x3(x)))
        if self.use_k5:
            outputs.append(torch.relu(self.conv5x5(x)))
        if self.use_k7:
            outputs.append(torch.relu(self.conv7x7(x)))

        x = torch.cat(outputs, dim=1)
        x = self.pool(x)
        return x


# In[ ]:


class SequentialCNNModel(nn.Module):
    def __init__(self, filters=[16,32,64,128,256],
                 dropout=0.3,
                 kernel_config=(True, True, True)):
        super().__init__()

        k3, k5, k7 = kernel_config

        self.model = nn.Sequential(
            ConcatConvBlock(93, filters[0], k3, k5, k7),
            ConcatConvBlock(filters[0]*sum(kernel_config), filters[1], k3, k5, k7),
            ConcatConvBlock(filters[1]*sum(kernel_config), filters[2], k3, k5, k7),
            ConcatConvBlock(filters[2]*sum(kernel_config), filters[3], k3, k5, k7),
            ConcatConvBlock(filters[3]*sum(kernel_config), filters[4], k3, k5, k7),

            nn.Flatten(),
            nn.Linear(filters[4]*sum(kernel_config)*7*15, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 250*500),
            nn.Identity()
        )

        self.reshape = (250, 500)

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, *self.reshape)


# In[15]:


# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[16]:


model_37 = SequentialCNNModel(kernel_config=(True, False, True))

model = model_37.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)
criterion = nn.MSELoss()


# In[17]:


epochs = 200 # number of epochs
early_stopping_patience = 100
best_val_loss = float('inf')
patience_counter = 0
# Lists to store loss history
train_loss_history = []
val_loss_history = []


# In[18]:


for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_train_batch)
        loss = criterion(outputs, y_train_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    
    val_loss = 0.0
    if val_loader:
        model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f'{path}/best_model_weights.pth')
            torch.save(model, f'{path}/best_full_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    else:
        val_loss_history.append(None)  # If there's no validation loader, append None
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f} seconds")


# # Post processing

# In[ ]:


torch.save(model.state_dict(), f'{path}/model_weights.pth')


# In[ ]:


torch.save(model, f'{path}/full_model.pth')


# In[ ]:


# Save the final model and optimizer state
torch.save({
    'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_history': train_loss_history,
    'val_loss_history': val_loss_history,
}, f'{path}/model_checkpoint.pth')


# In[ ]:


# Postprocessing: Loading the checkpoint and plotting the loss history
checkpoint = torch.load(f'{path}/model_checkpoint.pth')
train_loss_history = checkpoint['train_loss_history']
val_loss_history = checkpoint['val_loss_history']


# In[ ]:


# Identify the epoch with the best model (lowest validation loss)
best_epoch = val_loss_history.index(min(val_loss_history)) + 1  # +1 to convert from 0-based to 1-based index
best_val_loss = min(val_loss_history)

# Print the best epoch and the corresponding validation loss
print(f'The best model is at epoch {best_epoch} with a validation loss of {best_val_loss:.4f}')

# Define the range of epochs
epochs = range(1, len(train_loss_history) + 1)

# Plot the training and validation loss
plt.plot(epochs, train_loss_history, label='Training Loss')
plt.plot(epochs, val_loss_history, label='Validation Loss')

# Highlight the best epoch
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot
plt.savefig(f'{path}/LossOverEpoch.png')

# Show the plot
plt.show()


# # Model Evaluation

# In[ ]:


model.load_state_dict(torch.load(f'{path}/best_model_weights.pth'))
model.eval()


# In[ ]:


# Evaluate the model on test data
criterion = nn.MSELoss()
test_loss = 0.0
all_predictions = []
all_targets = []


# In[ ]:


with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
        outputs = model(X_test_batch)
        loss = criterion(outputs, y_test_batch)
        test_loss += loss.item()
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(y_test_batch.cpu().numpy())


# In[ ]:


test_loss /= len(test_loader)
print(f"Test Loss (MSE): {test_loss:.4f}")


# In[ ]:


# Concatenate all predictions and targets
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)


# In[ ]:


# Flatten the arrays for MSE calculation
all_predictions_flat = all_predictions.flatten()
all_targets_flat = all_targets.flatten()


# In[ ]:


# Mean and std for de-normalization (your values)
mean_GRNHFX = 94.552284
std_GRNHFX  = 2804.3684

def de_normalize(data, mean, std):
    return (data * std) + mean

# ---- Flatten predictions and targets (these are STILL normalized) ----
pred_norm = all_predictions.reshape(-1)
targ_norm = all_targets.reshape(-1)

# ---- De-normalize to physical units ----
pred_phys = de_normalize(pred_norm, mean_GRNHFX, std_GRNHFX)
targ_phys = de_normalize(targ_norm, mean_GRNHFX, std_GRNHFX)

# ---- Define burning mask in PHYSICAL units ----
# Option A (strict): burning if target > 0
# burn_mask = targ_phys > 0.0

# Option B (recommended): use a small physical threshold to avoid tiny noise
# Choose based on your units. If GRNHFX is W/m², 1–10 is reasonable. If kW/m², use 0.001–0.01.
burn_threshold = 1e-10   # start with 0; change to e.g. 10.0 (W/m²) if needed
burn_mask = targ_phys > burn_threshold

# ---- Sanity checks ----
n_total = targ_phys.size
n_burn = int(np.sum(burn_mask))
print(f"Total pixels: {n_total:,}")
print(f"Burning pixels (target > {burn_threshold}): {n_burn:,} ({100*n_burn/n_total:.4f}%)")

if n_burn == 0:
    raise ValueError("No burning pixels found with the chosen threshold. "
                     "Increase/decrease 'burn_threshold' or inspect target values.")

# ---- Compute metrics on burning pixels only ----
targ_burn = targ_phys[burn_mask]
pred_burn = pred_phys[burn_mask]

mse_burn  = mean_squared_error(targ_burn, pred_burn)
rmse_burn = np.sqrt(mse_burn)
mae_burn  = mean_absolute_error(targ_burn, pred_burn)
r2_burn   = r2_score(targ_burn, pred_burn)

print("\n=== Metrics on burning area only (physical units) ===")
print(f"MSE: {mse_burn:.6g}")
print(f"RMSE: {rmse_burn:.6g}")
print(f"MAE : {mae_burn:.6g}")
print(f"R²  : {r2_burn:.6g}")

# ---- Optional: also compute global metrics (all pixels) for transparency ----
mse_all  = mean_squared_error(targ_phys, pred_phys)
rmse_all = np.sqrt(mse_all)
mae_all  = mean_absolute_error(targ_phys, pred_phys)
r2_all   = r2_score(targ_phys, pred_phys)

print("\n=== Metrics on ALL pixels (physical units, includes non-burning) ===")
print(f"MSE: {mse_all:.6g}")
print(f"RMSE: {rmse_all:.6g}")
print(f"MAE : {mae_all:.6g}")
print(f"R²  : {r2_all:.6g}")

# ---- Optional: save arrays for later analysis ----
np.save(f"{path}/test_predictions_phys.npy", pred_phys)
np.save(f"{path}/test_targets_phys.npy", targ_phys)
np.save(f"{path}/burn_mask_phys.npy", burn_mask)


# In[ ]:


print('whole code has run successfully')

