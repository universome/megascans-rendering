"""
This script prepares a NeRF synthetic dataset
First, we put images into a single dir

Then we extract angles from
and then creates a `dataset.json` file with these angles
"""

import os
import json
import argparse
from typing import Dict
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel

from utils import copy_files, store_to_zip, get_euler_angles, remove_root

#----------------------------------------------------------------------------

def prepare_for_training(src_dir: os.PathLike, dst_dir: os.PathLike, use_roll_angles: bool=False, save_as_zip: bool=False, num_jobs: int=8):
    ################################################
    # Step 1. Copying the images into a new folder #
    ################################################
    os.makedirs(dst_dir, exist_ok=True)
    filepaths = {os.path.relpath(os.path.join(root, fname), start=src_dir) for root, _dirs, files in os.walk(src_dir) for fname in files}
    filepaths = {f for f in filepaths if not (f.endswith('.json') or '_depth_' in f or '_normal_' in f)}
    filepaths = list(filepaths)
    copy_files(src_dir, filepaths, dst_dir, flatten_parent_dir=True, num_jobs=num_jobs)

    ###################################
    # Step 2. Processing the metadata #
    ###################################
    # Step 2.1. Loading the metadata
    metadata_files = [os.path.join(src_dir, d, 'metadata.json') for d in sorted(os.listdir(src_dir))]

    # Step 2.2. Combining into a single one
    transforms = {}
    for mfile in metadata_files:
        with open(mfile, 'r') as f:
            curr_collection_name = os.path.basename(os.path.dirname(mfile))
            curr_transforms = json.load(f)
            curr_transforms = {os.path.join(curr_collection_name, model_name, os.path.basename(t['file_path'])): t['transform_matrix'] for model_name in curr_transforms for t in curr_transforms[model_name]}
            transforms = {**curr_transforms, **transforms}

    # Step 2.3. Verifying it is correct.
    camera_angles = {f'{f}.png': get_euler_angles(np.array(t)) for f, t in transforms.items()}
    if not use_roll_angles:
        angles_values = np.array([v for v in camera_angles.values()])
        assert abs(angles_values[:, [2]]).mean() < 1e-5, f"The dataset contains roll angles: {abs(angles_values[:, 2]).sum()}."
        assert (angles_values[:, [0]] ** 2).sum() ** 0.5 > 0.1, "Broken yaw angles (all zeros)."
        assert (angles_values[:, [1]] ** 2).sum() ** 0.5 > 0.1, "Broken pitch angles (all zeros)."
        assert angles_values[:, [0]].min() >= -np.pi, f"Broken yaw angles (too small): {angles_values[:, [0]].min()}"
        assert angles_values[:, [0]].max() <= np.pi, f"Broken yaw angles (too large): {angles_values[:, [0]].max()}"
        assert angles_values[:, [1]].min() >= 0.0, f"Broken pitch angles (too small): {angles_values[:, [1]].min()}"
        assert angles_values[:, [1]].max() <= np.pi, f"Broken pitch angles (too large): {angles_values[:, [1]].max()}"
    origins = np.array([np.array(t)[:3, 3] for t in transforms.values()])
    distances = np.sqrt((origins ** 2).sum(axis=1))
    print(f'Mean/std for camera distance: {distances.mean()} {distances.std()}')
    new_metadata = {'camera_angles': camera_angles}

    # Step 2.4. Collecting class label information
    new_metadata['labels'] = collect_class_labels(dst_dir)

    # Step 2.5. Saving the metadata as `dataset.json` for EpiGRAF dataloader.
    with open(os.path.join(dst_dir, 'dataset.json'), 'w') as f:
        json.dump(new_metadata, f)

    #################################
    # Step 3. Compressing if needed #
    #################################
    if zip:
        print('Creating a zip archive (without compression)...')
        store_to_zip(dst_dir, delete_original=True)

#----------------------------------------------------------------------------

def read_transforms_from_file(file_path: os.PathLike) -> Dict:
    with open(file_path, 'r') as f:
        transforms = json.load(f)
        transforms = {x['file_path']: x['transform_matrix'] for x in transforms['frames']}

    return transforms

#----------------------------------------------------------------------------

def collect_class_labels(data_dir: os.PathLike) -> Dict:
    """
    Collect labels given a file structure
    We assume that a separate class is a separate src_dir of directories
    """
    assert os.path.isdir(data_dir), f"Not a src_dir: {data_dir}"
    # Step 1: collect locations for each image
    all_files = {os.path.relpath(os.path.join(root, fname), start=data_dir) for root, _dirs, files in os.walk(data_dir) for fname in files if not fname.endswith('.json')}
    file2parent = {f: os.path.dirname(f) for f in all_files}
    parent2id = {p: i for i, p in enumerate(set(file2parent.values()))}
    dataset_name = os.path.basename(data_dir)
    labels = {remove_root(f, dataset_name): parent2id[p] for f, p in file2parent.items()}

    print(f'Found {len(parent2id)} different class labels.')

    return labels

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--src_dir', type=str, help='Path to the NeRF scene src_dir')
    parser.add_argument('-t', '--dst_dir', type=str, help='Where to save the result?')
    parser.add_argument('--use_roll_angles', action='store_true', help='Should we estimate the roll angle as well?')
    parser.add_argument('--save_as_zip', action='store_true', help='Should we put the result into a zip archive?')
    parser.add_argument('--num_jobs', default=8, type=int, help='Number of parallel jobs when resizing the dataset.')
    args = parser.parse_args()

    prepare_for_training(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        use_roll_angles=args.use_roll_angles,
        save_as_zip=args.save_as_zip,
        num_jobs=args.num_jobs,
    )

#----------------------------------------------------------------------------
