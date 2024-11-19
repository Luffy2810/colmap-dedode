import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

from .utils1 import *
import h5py
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/luffy/continue/repos/DeDoDe')
from DeDoDe.utils import dual_softmax_matcher, to_pixel_coords, to_normalized_coords
from .ransac_1 import *
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params = [0,1568/2,784/2]

g8p = EightPointAlgorithmGeneralGeometry()
ransac = RANSAC_8PA()


class DualSoftMaxMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize=normalize,
                               inv_temp=inv_temp, threshold=threshold) 
                    for k_A, d_A, k_B, d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            return {
                'matches0': torch.cat([m['matches0'] for m in matches]),
                'matches1': torch.cat([m['matches1'] for m in matches]),
                'matching_scores0': torch.cat([m['matching_scores0'] for m in matches]),
                'matching_scores1': torch.cat([m['matching_scores1'] for m in matches])
            }
        
        # Calculate similarity matrix
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                               normalize=normalize, inv_temperature=inv_temp)
        
        B = keypoints_A.shape[0]  # batch size
        N_A = keypoints_A.shape[1]  # number of keypoints in A
        N_B = keypoints_B.shape[1]  # number of keypoints in B
        
        # Find mutual matches above threshold
        max_vals_A, max_idx_A = P.max(dim=-1)  # Best matches in B for each point in A
        max_vals_B, max_idx_B = P.max(dim=-2)  # Best matches in A for each point in B
        
        # Mutual matches: points that chose each other as best matches
        mutual_matches = max_idx_B[torch.arange(B, device=P.device)[:, None], max_idx_A] == torch.arange(N_A, device=P.device)[None, :]
        above_threshold = max_vals_A > threshold
        valid_matches = mutual_matches & above_threshold
        
        # For each batch, apply RANSAC
        matches_list = []
        scores_list = []
        matches0 = torch.full((B, N_A), -1, dtype=torch.long, device=P.device)
        matches1 = torch.full((B, N_B), -1, dtype=torch.long, device=P.device)
        matching_scores0 = torch.zeros((B, N_A), dtype=torch.float32, device=P.device)
        matching_scores1 = torch.zeros((B, N_B), dtype=torch.float32, device=P.device)
        
        for b in range(B):
            # Get valid matches for this batch
            batch_valid = valid_matches[b]
            if batch_valid.sum() < 8:  # Need at least 8 points for RANSAC
                continue
                
            # Get matching points
            idx_A = torch.where(batch_valid)[0]
            idx_B = max_idx_A[b, batch_valid]
            
            # Convert points to spherical coordinates and apply RANSAC
            points_A = keypoints_A[b, batch_valid].cpu().numpy()
            points_B = keypoints_B[b, idx_B].cpu().numpy()
            
            points0_spherical = cam_from_img_vectorized(params, points_A)
            points1_spherical = cam_from_img_vectorized(params, points_B)
            
            try:
                ransac_inliers = ransac.get_inliers(points0_spherical.T, points1_spherical.T)[0]
                ransac.reset()
                
                if ransac_inliers.sum() > 0:
                    # Get the indices of inlier matches
                    inlier_idx_A = idx_A[ransac_inliers]
                    inlier_idx_B = idx_B[ransac_inliers]
                    
                    # Update matches and scores
                    matches0[b, inlier_idx_A] = inlier_idx_B
                    matches1[b, inlier_idx_B] = inlier_idx_A
                    
                    # Set matching scores for inliers
                    if P_A is not None and P_B is not None:
                        matching_scores0[b, inlier_idx_A] = P_A[b, inlier_idx_A]
                        matching_scores1[b, inlier_idx_B] = P_B[b, inlier_idx_B]
                    else:
                        scores = P[b, inlier_idx_A, inlier_idx_B]
                        matching_scores0[b, inlier_idx_A] = scores
                        matching_scores1[b, inlier_idx_B] = scores
            
            except Exception as e:
                logger.warning(f"RANSAC failed for batch {b}: {str(e)}")
                continue
        print ((matches0 != -1).sum().item())
        return {
            'matches0': matches0,  # (B, N_A) containing indices of matches in B (-1 for no match)
            'matches1': matches1,  # (B, N_B) containing indices of matches in A (-1 for no match)
            'matching_scores0': matching_scores0,  # (B, N_A) containing matching scores
            'matching_scores1': matching_scores1   # (B, N_B) containing matching scores
        }

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)

class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs

@torch.no_grad()
def match_from_paths(
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
) -> Path:
    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        print("Skipping the matching.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualSoftMaxMatcher().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model.match(
            data["keypoints0"], data["descriptors0"],
            data["keypoints1"], data["descriptors1"],
            P_A=data.get("scores0"), P_B=data.get("scores1"),
            normalize=True, inv_temp=20, threshold=0.01
        )
        
        
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    print("Finished exporting matches.")


def main(
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(pairs, matches, features_q, features_ref, overwrite)

    return matches


if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    pairs_path = Path("/home/luffy/continue/repos/COLMAP-ELoFTR/image_pairs_1.txt")
    features = Path("/home/luffy/continue/repos/DeDoDe/output/features.h5")
    matches = Path("/home/luffy/continue/repos/DeDoDe/output/matches.h5")

    matches_path = main(
        pairs=pairs_path,
        features=features,
        matches=matches,
        overwrite=False
    )