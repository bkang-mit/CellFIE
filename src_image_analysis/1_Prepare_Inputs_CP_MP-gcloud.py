import argparse
import os
import glob
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.filters import threshold_otsu, median
from skimage.morphology import disk
from imageio import volread as imread


def parse_args():
    parser = argparse.ArgumentParser(description="Process imaging data")
    parser.add_argument("--input_dir", required=True, help="Path to input directory")
    return parser.parse_args()


def create_directories(base_dir, dirs):
    for dir_name in dirs:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)


def background_subtract_2d(img, markers):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    for ch in range(4, img.shape[0]):
        if markers.loc[ch, 'excitation_wavelength'] == 488:
            img[ch] -= img[1]
        elif markers.loc[ch, 'excitation_wavelength'] == 561:
            img[ch] -= img[2]
        elif markers.loc[ch, 'excitation_wavelength'] == 637:
            img[ch] -= img[3]
        elif markers.loc[ch, 'excitation_wavelength'] == 514:
            img[ch] -= img[1]
        img[img < 0] = 0
    return img


def quantile_normalize(img, min_q, max_q):
    img_min = np.quantile(img, min_q, axis=(1, 2, 3), keepdims=True)
    img_max = np.quantile(img, max_q, axis=(1, 2, 3), keepdims=True)
    img = (img - img_min) / (img_max - img_min)
    return np.clip(img, 0, 1).astype(np.float32)


def prepare_mp_input(img, codebook, markers):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    img = img[markers['marker_name'].isin(codebook.index)]
    _img = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        for z in range(img.shape[1]):
            _img[c, z] = median(img[c, z], disk(1))
    return quantile_normalize(_img, 0.75, 0.999)


def matching_pursuit(x, A, max_iters, thr=1):
    A = A / np.linalg.norm(A, axis=1)[:, None]
    z = np.zeros((A.shape[0], *x.shape[1:]))
    x = x.copy()
    active_set = np.ones(x.shape[1:], dtype=bool)
    x_norm = np.linalg.norm(x, axis=0, ord=2)
    max_norm = threshold_otsu(x_norm) * thr
    active_set[x_norm <= max_norm] = False
    for _ in range(max_iters):
        Ax = A @ x[:, active_set]
        k_max = Ax.argmax(0)
        z[k_max, active_set] = Ax[k_max, range(len(k_max))]
        x[:, active_set] -= A[k_max].T * z[k_max, active_set]
        x_norm = np.linalg.norm(x, axis=0, ord=2)
        active_set[x_norm < max_norm] = False
    return z


def main():
    args = parse_args()
    data_dir = args.input_dir

    source = f'gs://fc-secure-9289bfef-e5cb-493a-83d5-e604cd429e39/Brian/{data_dir}/'
    max_dir, in_dir, mp_dir, mp_max_dir, cp_dir, meta_dir = 'max', 'merged', 'mp_score', 'mp_score_max', 'max_clean', 'metadata'
    dirs = [max_dir, in_dir, mp_dir, mp_max_dir, cp_dir]
    create_directories(data_dir, dirs)

    os.chdir(data_dir)
    full_codebook = pd.read_csv(f'../{meta_dir}/full_codebook.csv', index_col=0)
    procode_grna = pd.read_csv(f'../{meta_dir}/PROCODE_gRNA.csv')
    legal_codes = sorted(set(procode_grna['ProCode ID'].to_list()))
    codebook = full_codebook[legal_codes]
    markers = pd.read_csv(f'../{meta_dir}/markers.csv')
    
    # Extract FOVs
    cmd = f'gsutil -m rsync -r {source}reg_bin_Cyc_1 reg_bin_Cyc_1'
    os.system(cmd)
    all_fovs = sorted(glob.glob('reg_bin_Cyc_1/*'))
    fovs = [x.split('F')[-1][:3] for x in all_fovs]
    
    # Process each FOV in chunks
    for i in range(0, len(fovs), 10):
        for fov in fovs[i:i+10]:
            os.system(f'gsutil cp {source}{in_dir}/F{fov}.tif {in_dir}')
            img_merged = imread(f'{in_dir}/F{fov}.tif')
            img_max = img_merged.max(0)
            img_norm = quantile_normalize(background_subtract_2d(img_max, markers), 0.75, 0.9999)
            
            # Save images
            for dir_name, img, suffix in [(max_dir, img_max, "max"), (cp_dir, img_norm, "max_clean")]:
                tifffile.imwrite(f'./{dir_name}/F{fov}_{suffix}.tif', img.astype(np.float32), imagej=True, photometric='minisblack', metadata={'axes': 'CYX'})
                os.system(f'gsutil cp ./{dir_name}/F{fov}_{suffix}.tif {source}{dir_name}')

            # Matching Pursuit
            x = prepare_mp_input(img_merged, codebook, markers)
            z = matching_pursuit(x, codebook.values.T, max_iters=3, thr=0.25).clip(0, np.inf).astype(np.float32).transpose(1, 0, 2, 3)
            
            # Save MP results
            for dir_name, img, suffix in [(mp_dir, z, "mp_score"), (mp_max_dir, z.max(0), "mp_score_max")]:
                tifffile.imwrite(f'./{dir_name}/F{fov}_{suffix}.tif', img, imagej=True, photometric='minisblack', metadata={'axes': 'CYX' if "max" in suffix else 'ZCYX'})
                os.system(f'gsutil cp ./{dir_name}/F{fov}_{suffix}.tif {source}{dir_name}')
            
            del img_merged, img_max, img_norm, x, z
        
        # Clean up intermediate files
        for dir_name in [mp_dir, mp_max_dir, cp_dir, max_dir, in_dir]:
            os.system(f'rm -rf {dir_name}/*')


if __name__ == "__main__":
    main()
