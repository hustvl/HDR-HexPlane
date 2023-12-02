import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import get_ray_directions_blender, get_rays, read_pfm
from .colmap_dataset import *
class DoubleColmapDataset(Dataset):
    def __init__(
        self,
        datadir1,
        datadir2,
        split="train",
        downsample=2.0,
        is_stack=False,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
    ):
        self.dataset1 = ColmapDataset(datadir1,
                                      split,
                                      downsample,
                                      is_stack, cal_fine_bbox, N_vis,
                                      time_scale, scene_bbox_min,
                                      scene_bbox_max, N_random_pose)
        self.dataset2 = ColmapDataset(datadir2,
                                      split,
                                      downsample,
                                      is_stack, cal_fine_bbox, N_vis,
                                      time_scale, scene_bbox_min,
                                      scene_bbox_max, N_random_pose)
        self.ndc_ray = self.dataset2.ndc_ray
        self.white_bg = self.dataset2.white_bg
        self.near_far = self.dataset1.near_far
        self.split = split
        self.downsample = downsample
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )


        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)


        self.white_bg = True
        self.ndc_ray = False
        self.depth_data = False
        self.near_far = self.dataset1.near_far
        self.near, self.far = self.dataset1.near, self.dataset1.far
        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.W = self.dataset1.W + self.dataset2.W
        self.H = self.dataset1.H + self.dataset2.H
        if not self.is_stack:
            self.dataset2.all_idx += (len(self.dataset1.meta_train["frames"]) + len(self.dataset1.meta_test["frames"]))
            self.all_rgbs = torch.cat([self.dataset1.all_rgbs,self.dataset2.all_rgbs],0)
            self.all_rays = torch.cat([self.dataset1.all_rays,self.dataset2.all_rays],0)
            self.all_idx = torch.cat([self.dataset1.all_idx, self.dataset2.all_idx],0)
            self.all_times = torch.cat([self.dataset1.all_times, self.dataset2.all_times],0)
        else:

            self.dataset2.all_idx = [i + (len(self.dataset1.meta_train["frames"]) + len(self.dataset1.meta_test["frames"]))
                                         for i in self.dataset2.all_idx]
            self.all_rgbs = self.dataset1.all_rgbs + self.dataset2.all_rgbs
            self.all_rays = self.dataset1.all_rays + self.dataset2.all_rays
            self.all_idx = self.dataset1.all_idx + self.dataset2.all_idx
            self.all_times = self.dataset1.all_times + self.dataset2.all_times

    def get_total_image(self):
        return self.dataset1.get_total_image() + self.dataset2.get_total_image()

    def compute_bbox(self):
        xyzmin_1, xyzmax_1 = self.dataset1.compute_bbox()
        xyzmin_2, xyzmax_2 = self.dataset2.compute_bbox()
        xyz_min = torch.min(xyzmin_1, xyzmin_2)
        xyz_max = torch.max(xyzmax_1, xyzmax_2)
        return xyz_min, xyz_max


    def __len__(self):
        return len(self.all_rgbs)

    def get_video_split_index(self):
        index = [0 for i in range(len(self.dataset1.meta_train["frames"]) + len(self.dataset1.meta_test["frames"]))] + \
                [1 for i in range(len(self.dataset1.meta_train["frames"]) + len(self.dataset1.meta_test["frames"]))]
        return torch.tensor(index)
    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        rays_all_1, times_val_1, idxs_val_1 = self.dataset1.get_val_rays()
        rays_all_2, times_val_2, idxs_val_2 = self.dataset2.get_val_rays()
        # 还需要确定index。
        # val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = torch.cat([rays_all_1, rays_all_2], 0)
        val_times = times_val_1 + times_val_2
        idxs_val = torch.cat([idxs_val_1, idxs_val_2], 0)
        return rays_all, torch.FloatTensor(val_times), idxs_val

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "time": self.all_times[idx],
                "idx": self.all_idx[idx]
            }
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            imageidx = self.all_idx[idx]
            sample = {"rays": rays, "rgbs": img, "time": time, "idx": imageidx}
        return sample
