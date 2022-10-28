import os
import shutil
import contextlib
import zipfile
from typing import List, Tuple

import joblib
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm

#----------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

#----------------------------------------------------------------------------

def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()

#----------------------------------------------------------------------------

def store_to_zip(dir_to_compress: os.PathLike, delete_original: bool=False):
    shutil.make_archive(dir_to_compress, 'zip', root_dir=os.path.dirname(dir_to_compress), base_dir=os.path.basename(dir_to_compress))

    if delete_original:
        shutil.rmtree(dir_to_compress)

#----------------------------------------------------------------------------

def listdir_full_paths(d: os.PathLike) -> List[os.PathLike]:
    return [os.path.join(d, o) for o in sorted(os.listdir(d))]

#----------------------------------------------------------------------------

def get_euler_angles(T: np.ndarray) -> Tuple[float, float, float]:
    yaw = np.arctan2(T[1, 0], T[0, 0]).item()
    pitch = np.arctan2(T[2, 1], T[2, 2]).item()
    if pitch < 0:
        assert pitch < 1e-8 or (np.pi + pitch) < 1e-8, f"Cannot handle pitch value: {pitch}"
        pitch = abs(pitch)
    roll = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2)).item()

    return yaw, pitch, roll

#----------------------------------------------------------------------------

def copy_files(src_path: os.PathLike, files_to_copy: List[os.PathLike], dst_path: os.PathLike, flatten_parent_dir: bool=False, num_jobs: int=8):
    """
    src_path --- main dataset directory
    files_to_copy --- filepaths inside the `src_path` directory
    dst_path --- where to save the files
    """
    jobs = []
    dirs_to_create = []

    for filepath in tqdm(files_to_copy, desc='Collecting jobs'):
        src_file_path = os.path.join(src_path, filepath)
        if flatten_parent_dir:
            parent_dir = os.path.dirname(filepath)
            dst_file_path = os.path.join(dst_path, os.path.dirname(parent_dir), f'{os.path.basename(parent_dir)}_{os.path.basename(filepath)}')
        else:
            dst_file_path = os.path.join(dst_path, filepath)
        dirs_to_create.append(os.path.dirname(dst_file_path))
        jobs.append(delayed(shutil.copy)(src=src_file_path, dst=dst_file_path))

    for d in tqdm(list(set(dirs_to_create)), desc='Creating necessary directories'):
        if d != '':
            os.makedirs(d, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=num_jobs)(jobs)

#----------------------------------------------------------------------------

def remove_root(fname: os.PathLike, root_name: os.PathLike):
    """`root_name` should NOT start with '/'"""
    if fname == root_name or fname == ('/' + root_name):
        return ''
    elif fname.startswith(root_name + '/'):
        return fname[len(root_name) + 1:]
    elif fname.startswith('/' + root_name + '/'):
        return fname[len(root_name) + 2:]
    else:
        return fname

#----------------------------------------------------------------------------
