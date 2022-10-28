import os
import json
import shutil
import ctypes
import argparse
import contextlib
import multiprocessing as mp
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from utils import tqdm_joblib, listdir_full_paths

#----------------------------------------------------------------------------

def compute_opacities(src_path: os.PathLike):
    collection_paths = [d for d in listdir_full_paths(src_path) if os.path.isdir(d)]
    model_dirs = [d for c in collection_paths for d in listdir_full_paths(c) if os.path.isdir(d)]
    manager = mp.Manager()
    storage = manager.list([0.0] * len(model_dirs))
    jobs = []

    for i, m in enumerate(model_dirs):
        jobs.append(delayed(compute_average_opacity)(model_dir=m, storage=storage, index=i))

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=32)(jobs)

    return {m: storage[i] for i, m in enumerate(model_dirs)}

#----------------------------------------------------------------------------

def compute_average_opacity(model_dir: os.PathLike, storage: List, index: int):
    img_paths = [p for p in listdir_full_paths(model_dir) if p.endswith('.png') and not 'normal' in p]
    imgs = np.array([np.array(Image.open(p)) for p in img_paths]).astype(float) / 255.0
    avg_opacity = imgs[:, :, :, 3].mean().item()
    storage[index] = avg_opacity

#----------------------------------------------------------------------------

def copy_collections(src_path: os.PathLike, models_to_copy: List[os.PathLike], trg_path: os.PathLike, num_jobs: int=8):
    """
    src_path --- main dataset directory
    models_to_copy --- model dir paths in the format "collection_name/model_name"
    trg_path --- where to save the filtered dataset
    """
    assert all([len(m.split('/')) == 2 for m in models_to_copy])
    col2models = {c: set() for c in set([m.split('/')[0] for m in models_to_copy])}
    for m in models_to_copy:
        c, m = m.split('/')
        col2models[c].add(m)

    jobs = []

    for collection_name, models in tqdm(list(col2models.items()), desc='Collecting jobs'):
        if len(models) == 0:
            assert False, f"How can there be an empty collection to copy?"
        os.makedirs(os.path.join(trg_path, collection_name), exist_ok=True)

        # First, filter and save the metadata
        with open(os.path.join(src_path, collection_name, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            new_metadata = {m: metadata[m] for m in metadata if m in models}
        with open(os.path.join(trg_path, collection_name, 'metadata.json'), 'w') as f:
            json.dump(new_metadata, f)

        # Next, create the copying jobs
        for model_name in models:
            jobs.append(delayed(shutil.copytree)(
                src=os.path.join(src_path, collection_name, model_name),
                dst=os.path.join(trg_path, collection_name, model_name)
            ))

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=num_jobs)(jobs)

#----------------------------------------------------------------------------

def filter_models(src_path: os.PathLike, trg_path: os.PathLike, opacity_remove_thresh: float):
    # Step 1. Compute the average opacity per model
    opacities = compute_opacities(src_path)
    # with open('tmp.json', 'w') as f:
    #     json.dump(opacities, f)
    # with open('tmp.json', 'r') as f:
    #     opacities = json.load(f)

    # Step 2. Filtering by opacity.
    models_to_copy = set([m for m in opacities if opacities[m] >= opacity_remove_thresh])
    models_to_ignore = set([m for m in opacities if not m in models_to_copy])
    keep_ratio = len(models_to_copy) / len(opacities)
    print(f'Ignoring {len(models_to_ignore)} models due to opacitiy. This is {100 - keep_ratio * 100:.02f}% of all the models. {len(models_to_copy)} models remain.')

    # Step 3. Remove just low-quality models (we manually inspected each model).
    with open('low-quality-models.txt', 'r') as f:
        low_quality_models = set([m for m in f.read().split('\n') if not m.startswith('#')])

    # Step 4. Filter the low-quality models.
    all_current_models = models_to_copy
    models_to_copy = set([m for m in all_current_models if not os.path.basename(m) in low_quality_models])
    models_to_ignore = set([m for m in all_current_models if not m in models_to_copy])
    keep_ratio = len(models_to_copy) / len(opacities)
    print(f'Ignoring {len(models_to_ignore)} more models due to low quality. This is {100 - keep_ratio * 100:.02f}% of all the models. {len(models_to_copy)} models remain.')

    # Step 5. Save the models.
    models_to_copy = [os.path.relpath(m, src_path) for m in models_to_copy]
    copy_collections(src_path, models_to_copy, trg_path)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_path', type=str, help='Path to the original directory with images.')
    parser.add_argument('-t', '--trg_path', type=str, help='Path to the target path where to save the results.')
    parser.add_argument('-r', '--opacity_remove_thresh', type=float, help='Remove all the models which have average opacity of less than `opacity_remove_thresh`.')
    args = parser.parse_args()

    filter_models(
        src_path=args.src_path,
        trg_path=args.trg_path,
        opacity_remove_thresh=args.opacity_remove_thresh,
    )

#----------------------------------------------------------------------------
