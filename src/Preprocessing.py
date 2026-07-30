#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
#import matplotlib.pyplot as plt
import pylab as pl
import math
import netCDF4 as nc
import os
from tqdm import tqdm
import glob
import dask
import dask.array as da
from dask_ml.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#%matplotlib inline


# In[2]:


os.chdir('/work/bb1070/b382483/Data/FRDR')


# In[3]:


dirpath = '/work/bb1070/b382483/Data/FRDR/wrfout_*'
       
dirlist = sorted(glob.glob(dirpath)) 


# In[4]:


exclude_runs_sim = ['W5F4R0','W5F9R1','W5F8R3','W5F9R3','W5F1R3','W5F13R0','W5F1R7T','W5F8R7T','W5F9R7T']


# In[5]:


exclude_runs_format = ['W11F7R7T', 'W5F7R6TE']
#W11F7R7T, W5F7R6TE


# In[6]:


#exclude_runs_seventy = ['W10F7R7T','W12F7R7T','W3F7R5TE','W3F7R6TE','W3F7R7T','W4F7R5TE',
#                      'W4F7R6TE','W4F7R7T','W5F10R7T','W5F11R7T','W5F12R5TE','W5F12R6TE',
#                       'W5F12R7T','W5F13R5TE','W5F13R6TE', 'W5F13R7T','W5F1R3','W5F1R7T',
#                       'W5F2R7T', 'W5F3R7T','W5F4R5TE', 'W5F4R6TE','W5F4R7T','W5F5R7T', 
#                       'W5F5R7T', 'W5F6R5TE','W5F6R6TE','W5F6R7T', 'W5F7R5TE','W5F7R7T',
#                        'W5F7R8T','W6F7R7T', 'W7F7R7T', 'W8F7R7T','W9F7R7T']
    
exclude_runs_seventy =['W3F7R5TE','W3F7R6TE','W4F7R5TE','W4F7R6TE','W5F12R5TE','W5F12R6TE','W5F13R5TE',
                       'W5F13R6TE','W5F4R5TE','W5F4R6TE','W5F6R5TE', 'W5F6R6TE', 'W5F7R5TE']


# In[7]:


exclude_runs= exclude_runs_sim+ exclude_runs_format+exclude_runs_seventy


# In[31]:


datasets = [xr.open_mfdataset(filename, engine='netcdf4', parallel=True,combine='by_coords').isel(bottom_top=10, bottom_top_stag=10)
            for filename in dirlist if os.path.basename(filename).split('_')[-1] not in exclude_runs]


# # Preprocessing

# ### Min-Max Normalization 
# was performed for each feature individually according to equation (1) to quantify the variables and transform the original variable ranges into new
# ranges.
# 
# Zi=(xi-min i)/()max i-min i)
# 
# where Zi is the output value, xi is the value of the variables, and mean represents the average value of the variable and σi is the standard deviation of the variable.

# In[9]:


variables = ['U', 'V', 'W', 'T', 'QVAPOR', 'GRNHFX']
#chunk_sizes = {'U': (5, 250, 501), 'V': (5, 251, 500), 'W': (5, 250, 500), 'T': (5, 250, 500), 'QVAPOR': (5, 250, 500), 'GRNHFX': (5, 250, 500)}


# In[10]:


# Convert the data to Dask arrays and stack them
#data = {var: da.stack([da.from_array(ds[var].values, chunks=chunk_sizes[var]) for ds in datasets], axis=0) for var in variables}
# Stack the arrays
data = {var: np.stack([ds[var].values for ds in datasets], axis=0) for var in variables}


# In[11]:


def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


# In[12]:


# Apply Min-Max Normalization to each variable
for var in variables:
    #data[var] = normalize_data(data[var]).compute()
    data[var] = normalize_data(data[var])


# In[13]:


print (data['U'].shape)


# In[14]:


print (data['V'].shape)


# In[15]:


# Ignoring the last element along the third dimension for U_standardized
data['U'] = data['U'][:, :, :, :-1]


# In[16]:


# Ignoring the third dimension for V_standardized
data['V'] = data['V'][:, :, :-1, :]


# In[17]:


print (data['U'].shape)
print (data['V'].shape)


# In[18]:


#features = da.stack([data[var] for var in variables], axis=-1)
features = np.stack([data[var] for var in variables], axis=-1)
target = data['GRNHFX']


# In[19]:


print (features.shape)
print (target.shape)


# In[20]:


# Create lag features and targets for each time step
features_lagged = [feature[:-1, :, :] for feature in features]  # Lagged features (remove the last timestep)
target_lagged = [tar[1:, :, :] for tar in target]               # Lagged target (remove the first timestep)


# In[21]:


print (features_lagged[0].shape)
print (target_lagged[0].shape)


# In[22]:


# Convert lists to arrays
#datasets = da.stack(features_lagged, axis=0)
#labels = da.stack(target_lagged, axis=0)
datasets = np.stack(features_lagged, axis=0)
labels = np.stack(target_lagged, axis=0)


# In[23]:


print (datasets.shape)
print (labels.shape)


# In[24]:


# Assuming 'datasets' and 'labels' are your original data
num_samples = datasets.shape[0]
num_timesteps = datasets.shape[1]


# In[25]:


# Reshape the datasets
reshaped_datasets = datasets.reshape((num_samples * num_timesteps, 250, 500, 6))


# In[26]:


# Reshape the labels
reshaped_labels = labels.reshape((num_samples * num_timesteps, 250, 500))


# In[27]:


# Split datasets and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(reshaped_datasets, reshaped_labels, test_size=0.2, random_state=42)


# In[28]:


# Print the shapes of train and test sets
print("Train data shapes - X_train:", X_train.shape, "y_train:", y_train.shape)
print("Test data shapes - X_test:", X_test.shape, "y_test:", y_test.shape)


# In[29]:


# Convert Dask arrays to xarray DataArrays
X_train_da = xr.DataArray(X_train, name='X_train')
X_test_da = xr.DataArray(X_test, name='X_test')
y_train_da = xr.DataArray(y_train, name='y_train')
y_test_da = xr.DataArray(y_test, name='y_test')


# In[30]:


# Save xarray DataArrays as NetCDF files
X_train_da.to_netcdf('X_train.nc')
X_test_da.to_netcdf('X_test.nc')
y_train_da.to_netcdf('y_train.nc')
y_test_da.to_netcdf('y_test.nc')

