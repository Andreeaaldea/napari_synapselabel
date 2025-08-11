
# %%

import tifffile as tiff
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

# --- Config ---
registered_atlas_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\brainreg\registered_atlas.tiff'
high_res_raw_folder = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\StitchedImages_100\3'
output_mask_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\brainreg\mask_fullsize.tiff'

# --- Step 1: Load integer-labeled atlas ---
print("Loading registered_atlas.tiff...")
atlas_labels = tiff.imread(registered_atlas_path)  # Shape: (Z, Y, X)
print(f"Atlas shape: {atlas_labels.shape}")

# --- Step 2: Flip Z-axis (back to front) ---
atlas_labels = atlas_labels[::-1]
print("Flipping Z-axis (back to front)...")

# --- Step 3: Resize XY to match high-res image ---
raw_files = glob.glob(os.path.join(high_res_raw_folder, '*.tif')) + \
            glob.glob(os.path.join(high_res_raw_folder, '*.tiff'))
assert raw_files, "No TIFF files found in the high resolution folder."

high_res_example = tiff.imread(raw_files[0])
target_y, target_x = high_res_example.shape
print(f"Target XY shape: {(target_y, target_x)}")

# Resizing XY dimensions slice by slice (this will preserve regions correctly)
resized_xy_stack = np.empty((atlas_labels.shape[0], target_y, target_x), dtype=np.uint32)
print("Resizing XY dimensions slice by slice...")
for z in tqdm(range(atlas_labels.shape[0])):
    resized_slice = resize(
        atlas_labels[z].astype(float),
        (target_y, target_x),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint32)
    resized_xy_stack[z] = resized_slice

# --- Step 4: Resize Z dimension ---
#b118_z =2823
#b405_z= 2687
b396_z= 2487

target_z = b396_z
print(f"Resizing Z from {resized_xy_stack.shape[0]} to {target_z}")  # Set your desired target Z dimension here
resized_final = resize(
    resized_xy_stack.astype(float),
    (target_z, target_y, target_x),  # Resize the Z dimension along with XY
    order=0,
    preserve_range=True,
    anti_aliasing=False
).astype(np.uint32)

# --- Step 5: Flip X-axis (left-right) ---
final_mask = resized_final[:, :, ::-1]  # Flip the X-axis (left-right)
print(f"Final mask shape: {final_mask.shape}")

# --- Step 6: Save the final mask ---
print(f"Saving final resized region mask to {output_mask_path}...")
tiff.imwrite(output_mask_path, final_mask)
print("Done.")
# %%
import tifffile as tiff
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import glob

# --- Config ---
registered_atlas_path = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\WT_LC_2\brainreg\registered_atlas.tiff'
high_res_raw_folder = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\WT_LC_2\stitchedImages_100\3'
output_mask_path = r'\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\WT_LC_2\brainreg\mask_fullsize.tiff'

# --- Step 1: Load integer-labeled atlas ---
print("Loading registered_atlas.tiff...")
atlas_labels = tiff.imread(registered_atlas_path)  # Shape: (Z, Y, X)
print(f"Atlas shape: {atlas_labels.shape}")

# --- Step 2: Flip Z-axis (back to front) ---
atlas_labels = atlas_labels[::-1]
print("Flipping Z-axis (back to front)...")

# --- Step 3: Get target XY shape from example high-res image ---
raw_files = glob.glob(os.path.join(high_res_raw_folder, '*.tif')) + \
            glob.glob(os.path.join(high_res_raw_folder, '*.tiff'))
assert raw_files, "No TIFF files found in the high resolution folder."

high_res_example = tiff.imread(raw_files[0])
target_y, target_x = high_res_example.shape
print(f"Target XY shape: {(target_y, target_x)}")

# --- Step 4: Resize atlas to match high-res volume ---
b396_z = 2487  # Adjusted target Z
zoom_factors = (
    b396_z / atlas_labels.shape[0],     # Z
    target_y / atlas_labels.shape[1],   # Y
    target_x / atlas_labels.shape[2]    # X
)
print(f"Resizing atlas with zoom factors: {zoom_factors}...")
resized = zoom(atlas_labels, zoom=zoom_factors, order=0).astype(np.uint32)

# --- Step 5: Flip X-axis (left-right) ---
final_mask = resized[:, :, ::-1]
print(f"Final mask shape: {final_mask.shape}")

# --- Step 6: Save the final mask ---
print(f"Saving final resized region mask to {output_mask_path}...")
tiff.imwrite(output_mask_path, final_mask)


# %%
