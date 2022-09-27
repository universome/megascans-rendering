import os
import json
import shutil
import argparse
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm


#----------------------------------------------------------------------------

def remove_small_images(src_path: os.PathLike, trg_path: os.PathLike, opacity_remove_thresh: float):
    subdirs = [d for d in list_full_paths(src_path) if os.path.isdir(d)]
    if os.path.isfile(os.path.join(src_path, 'metadata.json')):
        model_paths = subdirs
    else:
        collection_paths = subdirs
        model_paths = [d for c in collection_paths for d in list_full_paths(c) if os.path.isdir(d)]
    # opacities = {}

    # for m in tqdm(model_paths):
    #     img_paths = [p for p in list_full_paths(m) if p.endswith('.png') and not 'normal' in p]
    #     imgs = np.array([np.array(Image.open(p)) for p in img_paths]).astype(float) / 255.0
    #     mean_opcaitiy = imgs[:, :, :, 3].mean()
    #     opacities[m] = mean_opcaitiy

    # with open('tmp2.json', 'w') as f:
    #     json.dump(opacities, f)



    with open('tmp.json', 'r') as f:
        opacities = json.load(f)

    vals = np.array(list(opacities.values()))
    vals_to_keep = vals[vals > opacity_remove_thresh]
    print(f'preserving {len(vals_to_keep) / len(vals): 0.2f}%:', len(vals_to_keep), 'out of', len(vals))

#----------------------------------------------------------------------------

def list_full_paths(dir_path: os.PathLike) -> List[os.PathLike]:
    """
    Returns a list of full paths to all objects in the given directory.
    """
    return [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))]

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_path', type=str, help='Path to the original directory with images.')
    parser.add_argument('-t', '--trg_path', type=str, help='Path to the target path where to save the results.')
    parser.add_argument('-r', '--opacity_remove_thresh', type=float, help='Remove all the models which have average opacity of less than `opacity_remove_thresh`.')
    args = parser.parse_args()

    remove_small_images(
        src_path=args.src_path,
        trg_path=args.trg_path,
        opacity_remove_thresh=args.opacity_remove_thresh,
    )

#----------------------------------------------------------------------------
