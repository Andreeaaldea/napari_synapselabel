
import numpy as np
import tifffile as tiff
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from skimage.morphology import white_tophat, disk
from scipy.ndimage import median_filter
import glob
import time

def compute_dff_folder(
    input_folder,
    output_folder,
    n_components=2,
    sample_fraction=0.1,
    denoise=True,
    background_radius=15,
    median_size=2,
    threshold=99504746098,
    verbose=True,
    test_mode=False,
    max_slices=None
):
    '''
    Compute dF/F for a folder of TIFF slices using a global Gaussian Mixture baseline.
    Applies optional background suppression and denoising.

    Parameters:
    - input_folder: folder of input .tiff slices (Z-stack)
    - output_folder: where to save dF/F .tif slices
    - n_components: number of Gaussians to fit (default=2)
    - sample_fraction: fraction of pixels to sample for baseline estimation
    - denoise: whether to apply white-tophat and median filter
    - background_radius: radius for white-tophat filter
    - median_size: size for median filtering
    - threshold: if set, zero out all values below this
    - verbose: whether to print progress
    - test_mode: process only a small number of slices for testing
    - max_slices: if set, process only this many slices (overrides test_mode)
    '''
    start_time = time.time()
    if verbose:
        print(f"\n[INFO] Processing: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    tiff_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")) +
                        glob.glob(os.path.join(input_folder, "*.tiff")))
    if verbose:
        print(f"Found {len(tiff_files)} TIFF files in: {input_folder}")

    if test_mode and max_slices is None:
        max_slices = 5
    if max_slices is not None:
        tiff_files = tiff_files[:max_slices]

    # --- Sample global pixel intensities to estimate F0 ---
    if verbose:
        print("Sampling voxel intensities for GMM...")
    intensities = []
    for f in tiff_files:
        img = tiff.imread(f).astype(np.float32)
        intensities.append(img.flatten())

    all_vals = np.concatenate(intensities)
    n_samples = int(len(all_vals) * sample_fraction)
    sample = np.random.choice(all_vals, n_samples, replace=False).reshape(-1, 1)

    if verbose:
        print("Fitting GMM...")
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(sample)
    f0 = np.sort(gmm.means_.flatten())[0]
    if verbose:
        print(f"Estimated F₀: {f0:.3f}")

    # --- Process each slice ---
    if verbose:
        print("Processing slices and writing dF/F outputs...")
    selem = disk(background_radius)

    for f in tiff_files:
        img = tiff.imread(f).astype(np.float32)
        dff = (img - f0) / (f0 + 1e-8)

        if denoise:
            dff = white_tophat(dff, footprint=selem)
            dff = median_filter(dff, size=median_size)

        if threshold is not None:
            dff = np.where(dff > threshold, dff, 0)

        out_path = os.path.join(output_folder, os.path.basename(f))
        tiff.imwrite(out_path, dff.astype(np.float32))

    elapsed = time.time() - start_time
    if verbose:
        print(f"[INFO] Finished {input_folder} in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

