
# %%
import numpy as np
import pandas as pd
import tifffile as tiff

# --- File paths ---
# Path to 3D mask TIFF file (each voxel is a region label)
mask_file = '/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/LC/A065/181_1103/brainreg/mask_fullsize.tiff'
# Path to 3D background-subtracted green channel TIFF file
signal_file = '/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/LC/A065/181_1103/brainreg/synapses_no_bg.tiff'

# --- Load the Data ---
mask = tiff.imread(mask_file)       # shape: (z, y, x)
green_img = tiff.imread(green_file)   # should have the same shape as mask

# --- Region-Based Intensity Calculation ---
results = []

# Get all unique region labels (skip background assumed to be 0)
region_labels = np.unique(mask)
region_labels = region_labels[region_labels != 0]



# %%

import tifffile as tiff
import numpy as np
import pandas as pd


# Path to the large TIFF file
mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\registered_atlas.tiff'
mask = tiff.imread(mask_file)  

# Count pixels per region
unique_labels, pixel_counts = np.unique(mask, return_counts=True)

# Create a dictionary of region labels and their pixel counts
region_pixel_counts = dict(zip(unique_labels, pixel_counts))
print(region_pixel_counts)


# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(region_pixel_counts.items()), columns=['Region', 'Pixel Count'])

# Save to CSV
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\region_pxs.csv'
df.to_csv(output_csv, index=False)
print(f"Saved pixel counts to {output_csv}")



# %%

for region in region_labels:
    # Create a boolean mask for the current region
    region_mask = (mask == region)
    
    # Calculate the mean and total (summed) intensity of green pixels in the region
    mean_intensity = np.mean(green_img[region_mask])
    total_intensity = np.sum(green_img[region_mask])
    
    # record the number of pixels (voxels) in this region
    pixel_count = np.sum(region_mask)
    
    results.append({
        'Region': region,
        'Mean_Intensity': mean_intensity,
        'Total_Intensity': total_intensity,
        'Pixel_Count': pixel_count
    })

# Convert results to a Pandas DataFrame for easier viewing and export
results_df = pd.DataFrame(results)
print(results_df)

# Save the results to a CSV file for later comparison with red cell counts
output_csv = '/ceph/mrsic_flogel/public/projects/AnAl_20240405_Neuromod_PE/brainsaw/LC/A065/181_1103/brainreg/region_syn_intensity.csv'
results_df.to_csv(output_csv, index=False)


#%%

import tifffile as tiff
import numpy as np
import pandas as pd

# --- File paths ---
# Path to 3D mask TIFF file (each voxel is a region label)
mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\registered_atlas.tiff'
# Path to 3D background-subtracted green channel TIFF file
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\downsampled.tiff'


# Open memory maps for both images. This creates array-like objects without reading everything at once.
mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

# --- Determine the set of regions ---
# Instead of loading all data to get unique labels, iterate slice by slice.
all_regions = set()
for z in range(mask.shape[0]):
    slice_labels = np.unique(mask[z])
    all_regions.update(slice_labels.tolist())
all_regions = np.array(sorted(list(all_regions)))

# Initialize dictionaries to accumulate intensities and pixel counts for each region.
region_total_intensity = {region: 0.0 for region in all_regions}
region_pixel_count = {region: 0 for region in all_regions}

# --- Process slice by slice ---
for z in range(mask.shape[0]):
    mask_slice = mask[z]
    green_slice = green_img[z]
    unique_regions_in_slice = np.unique(mask_slice)
    for region in unique_regions_in_slice:
        if region == 0:  # Skip background
            continue
        region_mask = (mask_slice == region)
        region_total_intensity[region] += np.sum(green_slice[region_mask])
        region_pixel_count[region] += np.sum(region_mask)

# --- Calculate mean intensities for each region ---
region_mean_intensity = {
    region: region_total_intensity[region] / region_pixel_count[region]
    for region in all_regions if region_pixel_count[region] > 0
}

# Create a DataFrame to organize the results.
results = pd.DataFrame({
    'Region': list(region_mean_intensity.keys()),
    'Mean_Intensity': list(region_mean_intensity.values()),
    'Total_Intensity': [region_total_intensity[region] for region in region_mean_intensity],
    'Pixel_Count': [region_pixel_count[region] for region in region_mean_intensity]
})

# Save the results to a CSV file.
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\downsampled_reg.csv'
results.to_csv(output_csv, index=False)
print(f"Region quantification results saved to {output_csv}")

# %%
import numpy as np
import pandas as pd
import tifffile as tiff

# Load the registered atlas TIFF file into a NumPy array.
# Each voxel's value corresponds to a region id.
atlas = tiff.imread(r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\registered_atlas.tiff')
print("Atlas shape:", atlas.shape)

# Load the structures.csv file that maps region ids to names.
# The CSV might have columns like 'id', 'name', etc.
structures = pd.read_csv(r'Y:\public\projects\AnAl_20240405_Neuromod_PE\code\cfos_preprocessing\allen_mouse_10um_v1.2\structures.csv')
print("Mapping preview:")
print(structures.head())

# Example: Get the region name for a specific region id.
for region_id in range(1,672):
    region_name = structures.loc[structures['id'] == region_id, 'name'].values
    if region_name.size > 0:
        print(f"Region id {region_id} corresponds to {region_name[0]}")
    else:
        print(f"Region id {region_id} not found in the mapping.")



# %%


import tifffile as tiff
import numpy as np
import pandas as pd
import glob

# --- File paths ---
# Path to 3D mask TIFF file (each voxel is a region label)
mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\brainreg\mask_fullsize.tiff'
# Path to 3D background-subtracted green channel TIFF file
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\processed_data\green_raw_HR.tiff'
# Path to structures.csv that maps region ids to region names
structures_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\code\cfos_preprocessing\allen_mouse_10um_v1.2\structures.csv'
# Output CSV file path
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\396_0304\brainreg\region_quantification_withnames.csv'

# --- Load the mapping file ---
# structures.csv is assumed to have at least 'id' and 'name' columns.
df_structures = pd.read_csv(structures_csv)
# Get the full list of region ids from the mapping file.
all_region_ids = df_structures['id'].unique()

# --- Open memory maps for both images ---
mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

# --- Initialize dictionaries using the full list of region IDs ---
# This ensures that every region defined in the atlas appears in the output.
region_total_intensity = {region: 0.0 for region in all_region_ids}
region_pixel_count = {region: 0 for region in all_region_ids}

# --- Process slice by slice ---
for z in range(mask.shape[0]):
    mask_slice = mask[z]
    green_slice = green_img[z]
    # Find the unique region ids in this slice
    unique_regions_in_slice = np.unique(mask_slice)
    for region in unique_regions_in_slice:
        # Skip background if background is labeled as 0
        if region == 0:
            continue
        # Only update if this region is defined in the atlas mapping
        if region in region_total_intensity:
            region_mask = (mask_slice == region)
            region_total_intensity[region] += np.sum(green_slice[region_mask])
            region_pixel_count[region] += np.sum(region_mask)
        # If the region is not in our mapping, you can choose to log it or ignore it.

# --- Calculate mean intensities for each region ---
region_mean_intensity = {}
for region in all_region_ids:
    if region_pixel_count[region] > 0:
        region_mean_intensity[region] = region_total_intensity[region] / region_pixel_count[region]
    else:
        # Assign NaN or 0 if no pixels are found; here we use NaN.
        region_mean_intensity[region] = np.nan

# --- Create a DataFrame with the quantification results ---
results = pd.DataFrame({
    'id': list(all_region_ids),
    'Mean_Intensity': [region_mean_intensity[r] for r in all_region_ids],
    'Total_Intensity': [region_total_intensity[r] for r in all_region_ids],
    'Pixel_Count': [region_pixel_count[r] for r in all_region_ids]
})

# --- Merge with the structures mapping to get region names ---
results = pd.merge(results, df_structures[['id', 'name']], on='id', how='left')
results.rename(columns={'name': 'Region_Name'}, inplace=True)

# --- Save the results to a CSV file ---
results.to_csv(output_csv, index=False)
print(f"Region quantification results saved to {output_csv}")


# %%


# %%
import tifffile as tiff
import numpy as np
import pandas as pd

# --- File paths ---
# Path to 3D mask TIFF file (each voxel is a region label)
mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\registered_atlas.tiff'
# Path to 3D background-subtracted green channel TIFF file
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\downsampled.tiff'
# Path to structures.csv that maps region ids to region names
structures_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\code\cfos_preprocessing\allen_mouse_10um_v1.2\structures.csv'
# Output CSV file path
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\LC\A065\181_1103\brainreg\downsampled_region_quantification_withnames.csv'

# --- Load the mapping file ---
# structures.csv is assumed to have at least 'id' and 'name' columns.
df_structures = pd.read_csv(structures_csv)
# Get the full list of region ids from the mapping file.
all_region_ids = df_structures['id'].unique()

# --- Open memory maps for both images ---
mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

# --- Initialize dictionaries using the full list of region IDs ---
# This ensures that every region defined in the atlas appears in the output.
region_total_intensity = {region: 0.0 for region in all_region_ids}
region_pixel_count = {region: 0 for region in all_region_ids}

# --- Process slice by slice ---
for z in range(mask.shape[0]):
    mask_slice = mask[z]
    green_slice = green_img[z]
    # Find the unique region ids in this slice
    unique_regions_in_slice = np.unique(mask_slice)
    for region in unique_regions_in_slice:
        # Skip background if background is labeled as 0
        if region == 0:
            continue
        # Only update if this region is defined in the atlas mapping
        if region in region_total_intensity:
            region_mask = (mask_slice == region)
            region_total_intensity[region] += np.sum(green_slice[region_mask])
            region_pixel_count[region] += np.sum(region_mask)

        # If the region is not in our mapping, you can choose to log it or ignore it.

# --- Calculate mean intensities for each region ---
region_mean_intensity = {}
for region in all_region_ids:
    if region_pixel_count[region] > 0:
        region_mean_intensity[region] = region_total_intensity[region] / region_pixel_count[region]
    else:
        # Assign NaN or 0 if no pixels are found; here we use NaN.
        region_mean_intensity[region] = np.nan

# --- Create a DataFrame with the quantification results ---
results = pd.DataFrame({
    'id': list(all_region_ids),
    'Mean_Intensity': [region_mean_intensity[r] for r in all_region_ids],
    'Total_Intensity': [region_total_intensity[r] for r in all_region_ids],
    'Pixel_Count': [region_pixel_count[r] for r in all_region_ids]
})

# --- Merge with the structures mapping to get region names ---
results = pd.merge(results, df_structures[['id', 'name']], on='id', how='left')
results.rename(columns={'name': 'Region_Name'}, inplace=True)

# --- Save the results to a CSV file ---
results.to_csv(output_csv, index=False)
print(f"Region quantification results saved to {output_csv}")

# %%
import numpy as np
import pandas as pd
import tifffile as tiff

mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\registered_atlas.tiff'
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\downsampled.tiff'
structures_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\code\cfos_preprocessing\allen_mouse_10um_v1.2\structures.csv'
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\preprocessed_data\region_pixel_intensities.csv'

df_structures = pd.read_csv(structures_csv)
all_region_ids = df_structures['id'].unique()

mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

mask_flat = mask.ravel()
green_flat = green_img.ravel()

region_pixel_intensities = {region: [] for region in all_region_ids}

for region in all_region_ids:
    idx = np.where(mask_flat == region)[0]
    if idx.size > 0:
        region_pixel_intensities[region] = green_flat[idx].tolist()
    else:
        region_pixel_intensities[region] = []

# Save as DataFrame (region id, pixel intensities as list)
df = pd.DataFrame({
    'id': list(region_pixel_intensities.keys()),
    'Pixel_Intensities': list(region_pixel_intensities.values())
})
df = pd.merge(df, df_structures[['id', 'name']], on='id', how='left')
df.rename(columns={'name': 'Region_Name'}, inplace=True)
df.to_csv(output_csv, index=False)
print(f"Pixel intensities for each region saved to {output_csv}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# Load mask and green channel (use memmap for large files)
mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\registered_atlas.tiff'
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\downsampled_channel_0.tiff'

mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

brain_pixels = []

for z in range(mask.shape[0]):
    mask_slice = mask[z]
    green_slice = green_img[z]
    brain_pixels.append(green_slice[mask_slice != 0])

# Concatenate all slices into one array
brain_pixels = np.concatenate(brain_pixels)


# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(brain_pixels, bins=100, color='green', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Distribution of Pixel Intensities Over Whole Brain')
plt.tight_layout()
plt.show()
# %%
import numpy as np
import pandas as pd
import tifffile as tiff

mask_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\registered_atlas.tiff'
green_file = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\registration\downsampled_channel_0.tiff'
structures_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\code\cfos_preprocessing\allen_mouse_10um_v1.2\structures.csv'
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\preprocessed_data\region_pixel_intensities.csv'

df_structures = pd.read_csv(structures_csv)
all_region_ids = df_structures['id'].unique()

mask = tiff.memmap(mask_file)
green_img = tiff.memmap(green_file)

mask_flat = mask.ravel()
green_flat = green_img.ravel()

region_pixel_intensities = {region: [] for region in all_region_ids}

for region in all_region_ids:
    idx = np.where(mask_flat == region)[0]
    if idx.size > 0:
        region_pixel_intensities[region] = green_flat[idx].tolist()
    else:
        region_pixel_intensities[region] = []

# Save as DataFrame (region id, pixel intensities as list)
df = pd.DataFrame({
    'id': list(region_pixel_intensities.keys()),
    'Pixel_Intensities': list(region_pixel_intensities.values())
})
df = pd.merge(df, df_structures[['id', 'name']], on='id', how='left')
df.rename(columns={'name': 'Region_Name'}, inplace=True)
df.to_csv(output_csv, index=False)
print(f"Pixel intensities for each region saved to {output_csv}")


# %%

# %%
output_csv = r'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping\Control\WT_Control_7\cellfinder\brainreg_gr_4\region_pixel_intensities.csv'

df.to_csv(output_csv, index=False)
print(f"Pixel intensities for each region saved to {output_csv}")
# %%
import pandas as pd
import ast
import matplotlib.pyplot as plt
# --- Convert stringified lists to actual lists ---
def parse_pixel_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return []


df['Pixel_Intensities'] = df['Pixel_Intensities'].map(parse_pixel_list)

# %%

# --- Flatten all pixel intensities ---
all_pixel_intensities = [pix for sublist in df['Pixel_Intensities'] for pix in sublist]



# %%

# Compute lower and upper bounds
lower = np.percentile(all_pixel_intensities, 1)   # 1st percentile
upper = np.percentile(all_pixel_intensities, 99)  # 99th percentile

# Filter the data
filtered_pixels = [x for x in all_pixel_intensities if lower <= x <= upper]
# %%
# --- Plot histogram ---
plt.figure(figsize=(10, 6))
plt.hist(all_pixel_intensities, bins=1000, color='green', alpha=0.75)
plt.xlim(0,1000)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Pixel Intensities Across All Regions")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
# %%


# --- Plot histogram ---
plt.figure(figsize=(10, 6))
plt.hist(all_pixel_intensities, bins=1000, color='green', alpha=0.75)
plt.xlim(1500,5000)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Pixel Intensities Across All Regions")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()