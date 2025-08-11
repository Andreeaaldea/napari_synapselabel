# %%

import numpy as np
from scipy.ndimage import gaussian_filter
import os
import glob
import numpy as np
import tifffile as tiff

# Define the folder containing your TIFF files
folder_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\stitchedImages_100\3'

# Get a sorted list of all TIFF files in the folder
tiff_files = sorted(glob.glob(os.path.join(folder_path, '*.tif')) + 
                    glob.glob(os.path.join(folder_path, '*.tiff')))

# Check if files were found
if not tiff_files:
    raise FileNotFoundError("No TIFF files found in the specified folder.")

# Read each TIFF file and stack them into a 3D NumPy array
stack = [tiff.imread(file) for file in tiff_files]
stack_3d = np.stack(stack, axis=0)

# Convert to float for processing
stack_bg_float = stack_3d.astype(float)

# Estimate the background with a Gaussian filter
sigma_value = 50  # Adjust this value as needed to capture the background scale
background = gaussian_filter(stack_bg_float, sigma=sigma_value)

# Subtract the background from the original image
bg_subtracted_stack = stack_bg_float - background
#cut 0, or negative values
#bg_subtracted_stack[bg_subtracted_stack < 0] = 0

# Convert back to the original data type (Tiff)
bg_subtracted_stack = bg_subtracted_stack.astype(stack_3d.dtype)

print("Background subtraction complete. Stack shape:", bg_subtracted_stack.shape)

# %%
import os
import glob
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

folder_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\stitchedImages_100\3'
tiff_files = sorted(glob.glob(os.path.join(folder_path, '*.tif')) +
                    glob.glob(os.path.join(folder_path, '*.tiff')))

if not tiff_files:
    raise FileNotFoundError("No TIFF files found in the specified folder.")

processed_slices = []

sigma_value = 50  # Adjust as needed

for file in tiff_files:
    # Load one slice
    slice_img = tiff.imread(file)
    
    # Convert to a lower-precision float to save memory (float32 instead of float64)
    slice_float = slice_img.astype(np.float32)
    
    # Estimate background using a Gaussian filter (2D)
    background = gaussian_filter(slice_float, sigma=sigma_value)
    
    # Subtract the background
    subtracted = slice_float - background
    
    # Clip negative values
    #subtracted[subtracted < 0] = 0
    
    # Convert back to original type and store
    processed_slices.append(subtracted.astype(slice_img.dtype))

# Stack the processed slices back into a 3D array
bg_subtracted_stack = np.stack(processed_slices, axis=0)
print("Background subtraction complete. Stack shape:", bg_subtracted_stack.shape)

# Save as a multi-page TIFF file. If the file is very large, you can enable BigTIFF.
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\green_no_bg_allval.tiff'
tiff.imwrite(output_file, bg_subtracted_stack, bigtiff=True)

print("Saved background subtracted stack with shape:", bg_subtracted_stack.shape)


# %%
import os
import glob
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

folder_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\stitchedImages_100\3'
tiff_files = sorted(glob.glob(os.path.join(folder_path, '*.tif')) +
                    glob.glob(os.path.join(folder_path, '*.tiff')))

if not tiff_files:
    raise FileNotFoundError("No TIFF files found in the specified folder.")

processed_slices = []

sigma_value = 50  # Adjust as needed

for file in tiff_files:
    # Load one slice
    slice_img = tiff.imread(file)
    
    # Convert to a lower-precision float to save memory (float32 instead of float64)
    slice_float = slice_img.astype(np.float32)
    
    # Estimate background using a Gaussian filter (2D)
    #background = gaussian_filter(slice_float, sigma=sigma_value)
    
    # Subtract the background
    subtracted = slice_float
    
    # Clip negative values
    #subtracted[subtracted < 0] = 0
    
    # Convert back to original type and store
    processed_slices.append(subtracted.astype(slice_img.dtype))

# Stack the processed slices back into a 3D array
bg_subtracted_stack = np.stack(processed_slices, axis=0)
print("Background subtraction complete. Stack shape:", bg_subtracted_stack.shape)

# Save as a multi-page TIFF file. If the file is very large, you can enable BigTIFF.
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\green_allval.tiff'
tiff.imwrite(output_file, bg_subtracted_stack, bigtiff=True)

print("Saved background subtracted stack with shape:", bg_subtracted_stack.shape)

# %%
#raw green channel data
import os
import numpy as np
import tifffile as tiff
from natsort import natsorted

# --- Path to the folder containing the TIFF slices ---
folder_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\stitchedImages_100\3'

# --- List all TIFF files in the folder ---
tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]

# --- Sort the files in natural order (1, 2, 3, ... instead of 1, 10, 2) ---
tiff_files = natsorted(tiff_files)

# --- Load all slices into a 3D NumPy array ---
print("Loading slices...")
slices = []

for file in tiff_files:
    file_path = os.path.join(folder_path, file)
    img = tiff.imread(file_path)
    
    # Ensure all slices have the same dimensions
    if len(slices) > 0 and img.shape != slices[0].shape:
        raise ValueError(f"Slice {file} has a different shape: {img.shape} vs {slices[0].shape}")
    
    slices.append(img)

# Stack into a 3D volume (Z, Y, X)
volume = np.stack(slices, axis=0)
print(f"Stacked volume shape: {volume.shape}")

# --- Save the 3D volume as a single TIFF file ---
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\processed_data\green_raw_HR.tiff'
print(f"Saving 3D volume to {output_file}...")

# Save the 3D stack as a multi-page TIFF
tiff.imwrite(output_file, volume, imagej=True)
# %%
