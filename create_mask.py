
# %%
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# Load the TIFF file
file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\registered_atlas.tiff'
print("Loading TIFF file...")
tiff_stack = tiff.imread(file)
print("TIFF file loaded.")

# Print the shape of the loaded stack
print(f"TIFF stack shape: {tiff_stack.shape}")

# Identify unique colors in the entire stack
print("Identifying unique colors...")
unique_colors = np.unique(tiff_stack)
print(f"Unique colors: {unique_colors}")

# Create a mapping from unique colors to numbers from 1 to 167
print("Creating color to number mapping...")
color_to_number = {color: i+1 for i, color in enumerate(unique_colors)}
print("Mapping created.")

# Function to apply the mapping to a slice
def map_colors(slice):
    return np.vectorize(color_to_number.get)(slice)

# Apply the mapping to the entire stack using parallel processing with a progress bar
print("Applying mapping to the stack...")
num_cores = -1  # Use all available cores
numbered_stack = Parallel(n_jobs=num_cores)(delayed(map_colors)(slice) for slice in tqdm(tiff_stack))
print("Mapping applied.")

# Convert the list of arrays back to a NumPy array
numbered_stack = np.array(numbered_stack)

# Print the shape of the numbered stack
print(f"Numbered stack shape: {numbered_stack.shape}")

# Display the first slice of the numbered stack
plt.imshow(numbered_stack[0], cmap='viridis')
plt.title('First Slice with Numbered Colors')
plt.colorbar()
plt.show()

# Save the numbered stack as a new TIFF file
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\numbered_stack.tiff'
print(f"Saving numbered stack to {output_file}...")
tiff.imwrite(output_file, numbered_stack)
print("Numbered stack saved.")

# %%
import os
import glob
import tifffile as tiff
from skimage.transform import resize
import numpy as np

# Flip the stack (back-to-front)
flipped_stack = numbered_stack[::-1]

# Define the folder containing the high-resolution raw data
raw_folder = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\stitchedImages_100\3'

# Search for TIFF files in that folder (you might need to adjust the extension if necessary)
raw_files = glob.glob(os.path.join(raw_folder, '*.tif')) + glob.glob(os.path.join(raw_folder, '*.tiff'))
if not raw_files:
    raise FileNotFoundError("No TIFF files found in the high resolution raw data folder.")

# Load one high-resolution image to get the target dimensions (height, width)
high_res_image = tiff.imread(raw_files[0])
target_shape = high_res_image.shape  # Assuming shape is (height, width)
print("Target shape (height, width):", target_shape)

# Create an empty array for the resized stack (same number of slices as flipped_stack)
resized_stack = np.empty((flipped_stack.shape[0], target_shape[0], target_shape[1]), dtype=flipped_stack.dtype)

# Resize each slice using nearest-neighbor interpolation (order=0)
for i in range(flipped_stack.shape[0]):
    resized_slice = resize(
        flipped_stack[i].astype(float),
        target_shape,
        order=0,                 # Nearest neighbor: keeps label values intact
        preserve_range=True,
        anti_aliasing=False
    ).astype(flipped_stack.dtype)
    resized_stack[i] = resized_slice

print("Resized stack shape:", resized_stack.shape)


# %%

#get slice number correct (for now check manually)
from skimage.transform import resize
current_z, current_y, current_x = resized_stack.shape
print(f"Current stack shape: {resized_stack.shape}")

# Define the target number of slices (z dimension)
target_z = 2823

# Define the new shape: (target_z, current_y, current_x)
new_shape = (target_z, current_y, current_x)

# Resize the entire stack using nearest neighbor interpolation (order=0)
# Preserve the original intensity range by using preserve_range=True
resized_stack_final = resize(
    resized_stack.astype(float),
    new_shape,
    order=0,                # Nearest neighbor interpolation
    preserve_range=True,
    anti_aliasing=False
).astype(resized_stack.dtype)

print("Resized and flipped stack saved.")
print("Final resized stack shape:", resized_stack_final.shape)

#%%

# Flip each slice horizontally by reversing the x-axis (axis=2)
flipped_lr_stack = resized_stack_final[:, :, ::-1]

# Alternatively, using np.flip:
# flipped_lr_stack = np.flip(resized_stack_final, axis=2)
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\mask_fullsize.tiff'
tiff.imwrite(output_file, flipped_lr_stack)
print("New shape after horizontal flip:", flipped_lr_stack.shape)



# %%
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# Load the TIFF file
file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\registered_atlas.tiff'
print("Loading TIFF file...")
tiff_stack = tiff.imread(file)
print("TIFF file loaded.")

# Print the shape of the loaded stack
print(f"TIFF stack shape: {tiff_stack.shape}")

# Identify unique colors in the entire stack
print("Identifying unique colors...")
unique_colors = np.unique(tiff_stack)
print(f"Unique colors: {unique_colors}")

# Create a mapping from unique colors to numbers from 1 to 167
print("Creating color to number mapping...")
color_to_number = {color: i+1 for i, color in enumerate(unique_colors)}
print("Mapping created.")

# Function to apply the mapping to a slice
def map_colors(slice):
    return np.vectorize(color_to_number.get)(slice)

# Define the range of slices to process (e.g., first 10 slices)
start_slice = 300
end_slice = 310
subset_tiff_stack = tiff_stack[start_slice:end_slice]

# Apply the mapping to the subset of the stack using parallel processing with a progress bar
print(f"Applying mapping to slices {start_slice} to {end_slice}...")
num_cores = -1  # Use all available cores
numbered_subset_stack = Parallel(n_jobs=num_cores)(delayed(map_colors)(slice) for slice in tqdm(subset_tiff_stack))
print("Mapping applied to subset.")

# Convert the list of arrays back to a NumPy array
numbered_subset_stack = np.array(numbered_subset_stack)

# Print the shape of the numbered subset stack
print(f"Numbered subset stack shape: {numbered_subset_stack.shape}")

# Display the first slice of the numbered subset stack
plt.imshow(numbered_subset_stack[0], cmap='viridis')
plt.title('First Slice with Numbered Colors')
plt.colorbar()
plt.show()

# Save the numbered subset stack as a new TIFF file
output_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\numbered_subset_stack.tiff'
print(f"Saving numbered subset stack to {output_file}...")
tiff.imwrite(output_file, numbered_subset_stack)
print("Numbered subset stack saved.")

# %%