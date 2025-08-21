
import os
import glob
from multiprocessing import Pool, cpu_count
from compute_dff_folder import compute_dff_folder

def process_single_brain(args):
    input_folder, output_folder = args
    compute_dff_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        n_components=2,
        sample_fraction=0.05,
        denoise=True,
        background_radius=20,
        median_size=2,
        threshold=0.1
    )

def batch_process_brains(base_input_dir, base_output_dir):
    """
    Processes all brain folders inside base_input_dir.
    Each subfolder is assumed to contain TIFF slices.
    """
    brain_folders = sorted([d for d in glob.glob(os.path.join(base_input_dir, "*")) if os.path.isdir(d)])
    print(f"Found {len(brain_folders)} brains.")

    args_list = []
    for brain_path in brain_folders:
        brain_name = os.path.basename(brain_path)
        output_path = os.path.join(base_output_dir, brain_name)
        os.makedirs(output_path, exist_ok=True)
        args_list.append((brain_path, output_path))

    # Run in parallel across brains
    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        pool.map(process_single_brain, args_list)

    print("✅ Batch processing complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Root directory containing all brain folders")
    parser.add_argument("--output_root", required=True, help="Root directory to save dF/F-normalized outputs")
    args = parser.parse_args()

    batch_process_brains(args.input_root, args.output_root)
