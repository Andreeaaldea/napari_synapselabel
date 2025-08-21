import os
import argparse
from multiprocessing import Pool, cpu_count
from compute_dff_folder import compute_dff_folder
from tqdm import tqdm
import fnmatch

def find_matching_folders(input_root, search_suffix):
    matching = []
    for root, dirs, _ in os.walk(input_root):
        for d in dirs:
            full_path = os.path.join(root, d)
            if fnmatch.fnmatch(full_path, f"*{search_suffix}"):
                matching.append(full_path)
    return matching

def process_folder(args):
    input_folder, output_folder, kwargs = args
    try:
        compute_dff_folder(input_folder, output_folder, **kwargs)
    except Exception as e:
        print(f"[ERROR] Skipping {input_folder} due to error: {e}")

def batch_process_recursive(input_root, output_root, search_suffix, test_mode=False):
    folders = find_matching_folders(input_root, search_suffix)
    if not folders:
        print("No matching folders found. Check your --input_root or folder structure.")
        return

    print(f"Found {len(folders)} matching folders.")

    args_list = []
    for input_folder in folders:
        rel_path = os.path.relpath(input_folder, input_root)
        out_folder = os.path.join(output_root, rel_path)
        kwargs = {
            'test_mode': test_mode,
            'max_slices': 5 if test_mode else None
        }
        args_list.append((input_folder, out_folder, kwargs))

    with Pool(processes=min(cpu_count(), len(args_list))) as pool:
        for _ in tqdm(pool.imap_unordered(process_folder, args_list), total=len(args_list)):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Root directory to search")
    parser.add_argument("--output_root", required=True, help="Root directory for outputs")
    parser.add_argument("--search_suffix", default="stitchedImages_100/2", help="Suffix to identify folders")
    parser.add_argument("--dry_run", action="store_true", help="Only list matching folders")
    parser.add_argument("--test", action="store_true", help="Run in test mode on a few slices")
    args = parser.parse_args()

    matching = find_matching_folders(args.input_root, args.search_suffix)
    if args.dry_run:
        print("Dry run: Matching folders")
        for folder in matching:
            print(folder)
    else:
        batch_process_recursive(args.input_root, args.output_root, args.search_suffix, test_mode=args.test)
