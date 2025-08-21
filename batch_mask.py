# process_mouse.py
import tifffile as tiff
import numpy as np
from scipy.ndimage import zoom
import os
import sys
import glob

# --- Mouse ID passed as argument ---
mouse_id = sys.argv[1]

# --- Path templates ---
base_dir = '/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/LC/A065/'
mouse_dir = os.path.join(base_dir, mouse_id)
registered_atlas_path = os.path.join(mouse_dir, 'brainreg', 'registered_atlas.tiff')
high_res_raw_folder = os.path.join(mouse_dir, 'data','stitchedImages_100', '3')
output_mask_path = os.path.join(mouse_dir, 'preprocessed_data', 'mask_fullsize.tiff')

print(f"Processing mouse {mouse_id}...")

# Load atlas
print("Loading registered_atlas.tiff...")
atlas_labels = tiff.imread(registered_atlas_path)
atlas_labels = atlas_labels[::-1]  # Flip Z

# Get XY shape
raw_files = glob.glob(os.path.join(high_res_raw_folder, '*.tif')) + \
            glob.glob(os.path.join(high_res_raw_folder, '*.tiff'))
assert raw_files, f"No TIFFs found in {high_res_raw_folder}"
high_res_example = tiff.imread(raw_files[0])
target_y, target_x = high_res_example.shape

# Resize
b396_z = 2487
zoom_factors = (
    b396_z / atlas_labels.shape[0],
    target_y / atlas_labels.shape[1],
    target_x / atlas_labels.shape[2]
)
resized = zoom(atlas_labels, zoom=zoom_factors, order=0).astype(np.uint32)
final_mask = resized[:, :, ::-1]

# Save
print(f"Saving final mask to {output_mask_path}...")
tiff.imwrite(output_mask_path, final_mask)
print("Done.")
