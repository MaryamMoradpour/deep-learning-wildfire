#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import glob
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio
from IPython.display import Image
from IPython.display import display


# In[2]:


# Define paths
path1 = "/work/bb1070/b382483/Data/TrainTest/ftp_stability/features_*"
path2 = "/work/bb1070/b382483/Data/TrainTest/ftp_stability/labels_*"
model_path = '/work/bb1070/b382483/Final/L=2km_new/NewCNNmodel/ftp_stability-identity/best_model_weights.pth'


# In[3]:


save_path = "/work/bb1070/b382483/Final/L=2km_new/Plots/TimeError/Identity"

# Create a new, empty directory
os.makedirs(save_path, exist_ok=True)


# In[4]:


# Mean and std for de-normalization (replace with actual values)
mean_GRNHFX = 94.552284  # Replace with the actual mean
std_GRNHFX = 2804.3684   # Replace with the actual std dev


# In[5]:


# Function to de-normalize data
def de_normalize(data, mean, std):
    return (data * std) + mean


# In[6]:


# Function to load datasets
def load_datasets(file_index, files1, files2):
    feature_file = files1[file_index]
    label_file = files2[file_index]
    ds_feature = xr.open_dataset(feature_file)
    ds_label = xr.open_dataset(label_file)
    return ds_feature, ds_label


# In[7]:


class ConcatConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatConvBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = torch.relu(self.conv3x3(x))
        x2 = torch.relu(self.conv5x5(x))
        x3 = torch.relu(self.conv7x7(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pool(x)
        return x

class SequentialCNNModel(nn.Module):
    def __init__(self):
        super(SequentialCNNModel, self).__init__()
        self.model = nn.Sequential(
            ConcatConvBlock(93, 16),    # Block 1
            ConcatConvBlock(48, 32),    # Block 2
            ConcatConvBlock(96, 64),    # Block 3
            ConcatConvBlock(192, 128),  # Block 4
            ConcatConvBlock(384, 256),  # Block 5
            nn.Flatten(),               # Block 6
            nn.Linear(768 * 7 * 15, 128), # Block 7, adjusted based on the output of Block 5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),        # Block 7
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 250 * 500),  # Block 7
            #nn.ReLU()
            nn.Identity()
        )
        self.reshape = (250, 500)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, *self.reshape)
        return x


# In[8]:


# Function to generate predictions and de-normalize
def generate_predictions(model, ds_feature, mean_GRNHFX, std_GRNHFX, device):
    predictions = []
    with torch.no_grad():
        for time_step in range(len(ds_feature['Time'])):
            X = ds_feature.isel(Time=time_step).to_array().values
            if X.ndim == 4 and X.shape[0] == 1:
                X = X.squeeze(0)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            X_tensor = X_tensor.unsqueeze(0)  # Add batch dimension

            output = model(X_tensor)
            prediction = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

            # De-normalize the prediction
            prediction = de_normalize(prediction, mean_GRNHFX, std_GRNHFX)

            predictions.append(prediction)

    return predictions


# In[9]:


# ‑‑‑ METRIC UTILITIES ---------------------------------------------------------
def compute_time_series_metrics(label_da: xr.DataArray,
                                pred_da: xr.DataArray,
                                mask_zero: bool = True):
    """
    Returns two DataArrays (mae_t, rmse_t) with each Time slice reduced over space.
    """
    if mask_zero:
        mask = label_da != 0
        error = (pred_da - label_da).where(mask)
    else:
        error = pred_da - label_da

    mae_t  = np.abs(error).mean(dim=("south_north", "west_east"))
    rmse_t = np.sqrt((error ** 2).mean(dim=("south_north", "west_east")))
    return mae_t, rmse_t


# In[10]:


def plot_combined_metrics(
        mae_all, rmse_all, mae_noz, rmse_noz,
        idx, outdir,
        *,
        figsize=(10, 4),
        colors=("tab:red", "tab:blue"),
        y_label="Error (kW m⁻²)",
        title_prefix="File",
        ylim=None):
    """
    Plot MAE / RMSE time‑series for one file (or for the pooled “ALL” case).

    Parameters
    ----------
    mae_all, rmse_all, mae_noz, rmse_noz : xarray.DataArray
        Metric series produced by `compute_time_series_metrics`.
    idx : int | str
        File number (e.g. 27) **or** a tag such as `"ALL"`.
    outdir : str | Path
        Folder where the PNG will be saved.
    figsize, colors, y_label, title_prefix, ylim : optional
        Styling knobs; change as needed.
    """
    # ------------------------------------------------------------------
    # 1.  House‑keeping
    # ------------------------------------------------------------------
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag     = f"{idx:03d}" if isinstance(idx, int) else str(idx)
    pngpath = outdir / f"errors_{tag}_combined.png"

    # ------------------------------------------------------------------
    # 2.  Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # solid = all cells  |  dashed = zeros excluded
    ax.plot(mae_all.Time,  mae_all,
            color=colors[0], linestyle="-",  label="MAE (all)")
    ax.plot(rmse_all.Time, rmse_all,
            color=colors[1], linestyle="-",  label="RMSE (all)")

    ax.plot(mae_noz.Time,  mae_noz,
            color=colors[0], linestyle="--", label="MAE (no‑zero)")
    ax.plot(rmse_noz.Time, rmse_noz,
            color=colors[1], linestyle="--", label="RMSE (no‑zero)")

    ax.set_xlabel("Time step")           # or "Time" if you prefer
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} {tag} – MAE & RMSE")
    ax.grid(True, alpha=0.3)

    if ylim is not None:                 # keep option for fixed y‑axis
        ax.set_ylim(ylim)

    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(pngpath, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {pngpath}")


# In[ ]:


# --------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------
def main():
    # ----------------------------------------------------------------
    # 0.  Paths and containers
    # ----------------------------------------------------------------
    files1 = sorted(glob.glob(path1))
    files2 = sorted(glob.glob(path2))

    mae_all_list, rmse_all_list   = [], []
    mae_noz_list, rmse_noz_list   = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SequentialCNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ----------------------------------------------------------------
    # 1.  Loop over files – keep per‑file metrics AND also store them
    # ----------------------------------------------------------------
    for idx in range(len(files1)):
        ds_feature, ds_label = load_datasets(idx, files1, files2)

        ds_label["GRNHFX"] = de_normalize(ds_label["GRNHFX"],
                                          mean_GRNHFX, std_GRNHFX)

        preds = generate_predictions(model, ds_feature,
                                     mean_GRNHFX, std_GRNHFX, device)

        preds_da = xr.DataArray(
            np.asarray(preds),
            dims  = ["Time", "south_north", "west_east"],
            coords = {"Time": ds_label["Time"],
                      "south_north": ds_label["south_north"] * 40,
                      "west_east"  : ds_label["west_east"]   * 40},
        )

        # kW m⁻²
        ds_label["GRNHFX"] /= 1000.0
        preds_da           /= 1000.0

        mae_all,  rmse_all  = compute_time_series_metrics(
                                  ds_label["GRNHFX"], preds_da, mask_zero=False)
        mae_noz, rmse_noz   = compute_time_series_metrics(
                                  ds_label["GRNHFX"], preds_da, mask_zero=True)

        # ---- keep for later aggregation ----
        mae_all_list.append(mae_all)
        rmse_all_list.append(rmse_all)
        mae_noz_list.append(mae_noz)
        rmse_noz_list.append(rmse_noz)

        # ---- per‑file CSV & plot (unchanged) ----
        #csv_path = os.path.join(save_path, f"errors_{idx:03d}.csv")
        #pd.DataFrame({
            #"Time"       : mae_all.Time.values,
            #"MAE_all"    : mae_all.values,
            #"RMSE_all"   : rmse_all.values,
            #"MAE_nozero" : mae_noz.values,
            #"RMSE_nozero": rmse_noz.values,
        #}).to_csv(csv_path, index=False)
        #plot_combined_metrics(mae_all, rmse_all, mae_noz, rmse_noz,
                             # idx, save_path)

    # ----------------------------------------------------------------
    # 2.  Aggregate across ALL files
    # ----------------------------------------------------------------
    # Stack along a new "file" dimension
    mae_all_stack = xr.concat(mae_all_list, dim="file")
    mae_noz_stack = xr.concat(mae_noz_list, dim="file")
    rmse_all_stack = xr.concat(rmse_all_list, dim="file")
    rmse_noz_stack = xr.concat(rmse_noz_list, dim="file")

    # Simple mean for MAE
    mae_all_mean = mae_all_stack.mean(dim="file")
    mae_noz_mean = mae_noz_stack.mean(dim="file")

    # Pooled RMSE = sqrt(mean(RMSE²))
    rmse_all_pooled = np.sqrt((rmse_all_stack ** 2).mean(dim="file"))
    rmse_noz_pooled = np.sqrt((rmse_noz_stack ** 2).mean(dim="file"))

    # ----------------------------------------------------------------
    # 3.  Save & plot the “all‑files” time‑series
    # ----------------------------------------------------------------
    csv_all = os.path.join(save_path, "errors_ALL_FILES.csv")
    pd.DataFrame({
        "Time"        : mae_all_mean.Time.values,
        "MAE_all"     : mae_all_mean.values,
        "RMSE_all"    : rmse_all_pooled.values,
        "MAE_nozero"  : mae_noz_mean.values,
        "RMSE_nozero" : rmse_noz_pooled.values,
    }).to_csv(csv_all, index=False)
    print(f"[CSV]  {csv_all}")

    plot_combined_metrics(mae_all_mean, rmse_all_pooled,
                          mae_noz_mean, rmse_noz_pooled,
                          "ALL", save_path)


# In[16]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




