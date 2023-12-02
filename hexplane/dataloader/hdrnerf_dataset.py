import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import get_ray_directions_blender, get_rays, read_pfm

blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w


class HDRnerfDataset(Dataset):
    def __init__(
        self,
        datadir,
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
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1


        self.define_transforms()  # transform to torch.Tensor

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.read_meta()  # Read meta data

        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        self.define_proj_mat()

        self.white_bg = True
        self.ndc_ray = False
        self.depth_data = False

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()
    def get_total_image(self):
        return len(self.image_paths)
    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            rays_o, rays_d = get_rays(self.directions, random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, *self.img_wh[::-1], 6
        )
    
        
    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        if isinstance(self.all_rays, list):
            rays = torch.cat([i.view(-1,6) for i in self.all_rays],0).view(-1,6)
            rays_o = rays[:,0:3]
            viewdirs = rays[:,3:6]
        else:
            rays_o = self.all_rays[:, 0:3]
            viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(self.root_dir, f"transforms_test.json")) as f:
            self.meta_test = json.load(f)
        with open(os.path.join(self.root_dir, f"transforms_train.json")) as f:
            self.meta_train = json.load(f)
        with open(os.path.join(self.root_dir, f"exposure_{self.split}.json")) as f:
            self.exps_meta = json.load(f)
        with open(os.path.join(self.root_dir, f"exposure_test.json")) as f:
            self.exps_test = json.load(f)
        with open(os.path.join(self.root_dir, f"exposure_train.json")) as f:
            self.exps_train = json.load(f)
            
        self.near = 0
        self.far = 15
        self.near_far = [self.near, self.far]
        w, h = self.img_wh

        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_blender(
            h, w, [self.focal, self.focal]
        )  # (h, w, 3)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )
        self.intrinsics = torch.tensor(
            [[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]
        ).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_times = []
        self.all_rgbs = []
        self.all_depth = []
        self.all_idx = []
        self.W = []
        self.H = []
        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))
        # read total time stamp
        self.test_timestamp = [0 for i in range(len(self.meta_test))]
        num_exps = 5
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#

            for idx in range(5):
                
                frame = self.meta["frames"][i]
                pose = np.array(frame["transform_matrix"])
                c2w = torch.FloatTensor(pose)
                self.poses += [c2w]
                path_tile = frame['file_path']
                
                image_path = os.path.join(self.root_dir, path_tile+'_%d.png'%idx)
                if not image_path.endswith("png"):
                    image_path+=".png"
                self.image_paths += [image_path]
                img = Image.open(image_path)
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                if self.split == "test":
                    img = img[:, :, w//2:]
                self.W.append(img.shape[2])
                self.H.append(img.shape[1])
                img = img.reshape(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (
                    1 - img[:, -1:]
                )  # blend A to RGB, white background

                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w, mode=self.split)  # Get rays, both (h*w, 3).
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                cur_time = torch.tensor(
                   [0]
                ).expand(rays_o.shape[0], 1)
                self.all_times += [cur_time]
                self.all_idx += [torch.tensor([i*5+idx]).expand(rays_o.shape[0], 1)]
                # print(i*5+idx,end=' ')

        if self.split == "train":
            # load left half image for training
            idxs = list(range(0,len(self.meta_test["frames"])))
            for i in tqdm(
                    idxs, desc=f"Loading data test_train ({len(idxs)})"
            ):  # img_list:#
                for idx in range(5):
                    frame = self.meta_test["frames"][i]
                    pose = np.array(frame["transform_matrix"])
                    c2w = torch.FloatTensor(pose)
                    self.poses += [c2w]
                    path_tile = frame['file_path']
                    image_path = os.path.join(self.root_dir, path_tile+'_%d.png'%idx)
                    self.image_paths += [image_path]
                    if not image_path.endswith("png"):
                        image_path += ".png"
                    img = Image.open(image_path)

                    if self.downsample != 1.0:
                        img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)  # (4, h, w)
                    img = img[:, :, :w // 2]
                    self.H.append(img.shape[1])
                    self.W.append(img.shape[2])
                    img = img.reshape(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                    img = img[:, :3] * img[:, -1:] + (
                            1 - img[:, -1:]
                    )  # blend A to RGB, white background

                    self.all_rgbs += [img]

                    rays_o, rays_d = get_rays(self.directions, c2w, mode="train_test")  # Get rays, both (h*w, 3).
                    self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
                    cur_time = torch.tensor( [0]
                    ).expand(rays_o.shape[0], 1)
                    self.all_times += [cur_time]
                    self.all_idx += [torch.tensor([i*5+idx]).expand(rays_o.shape[0], 1) + len(self.meta_train["frames"])*5]
                    # print(i*5+idx+len(self.meta["frames"])*5,end=' ')
        
        self.poses = torch.stack(self.poses)
        #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
        #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times, 0)
            self.all_idx = torch.cat(self.all_idx, 0)

            self.all_idx = torch.tensor(self.all_idx, dtype=torch.int32)
            print("all_idx:",self.all_idx.shape,self.all_idx.max(),self.all_idx.min())
            print("all_times:",self.all_times.shape)
            print("all_rays:",self.all_rays.shape)
            print("all_rgbs:",self.all_rgbs.shape)
        else:

            self.all_rays = [j.reshape(self.H[i], self.W[i], 6) for i, j in enumerate(self.all_rays)] # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = [j.reshape(self.H[i], self.W[i], 3) for i, j in enumerate(self.all_rgbs)]
            # self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
            #     -1, *self.img_wh[::-1], 3
            # )  # (len(self.meta['frames]),h,w,3)
            self.all_times = [j.reshape(self.H[i], self.W[i], 1) for i, j in enumerate(self.all_times)]
            self.all_times = [j for j in self.all_times]
            self.all_idx = [j.reshape(self.H[i], self.W[i], 1) for i, j in enumerate(self.all_idx)]

    def get_video_split_index(self):
        return torch.tensor([0 for i in range(len(self.meta_train["frames"])+len(self.meta_test["frames"]))])
        if "video" not in self.meta_train.keys():
            return torch.tensor([0 for i in range(len(self.meta_train["frames"])+len(self.meta_test["frames"]))])
        train_video = [0]+self.meta_train["video"]
        len_trainvideo = train_video[-1]
        test_video = [len_trainvideo]+[i+len_trainvideo for i in self.meta_test["video"]]
        video_split = [1 for i in range(train_video[-1])]

        len_testvideo = test_video[-1]
        cnt = 0
        for i in range(len(train_video)-1):
            video_split[train_video[i]:train_video[i+1]] = [cnt for i in range(train_video[i+1]-train_video[i])]
            video_split[test_video[i]:test_video[i+1]] =  [cnt for i in range(test_video[i+1]-test_video[i])]
            cnt+=1
        print("videosplit: ",video_split)
        return torch.tensor(video_split)
    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )

        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

        
    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        # val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        val_poses = self.poses
        val_times = self.test_timestamp
        rays_all = []  # initialize list to store [rays_o, rays_d]
        val_idxs = torch.Tensor([i+len(self.meta_train["frames"]) for i in range(len(self.meta_test["frames"]))])

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times), val_idxs
    def get_path_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        # val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        val_poses = self.poses
        val_times = self.test_timestamp 
        rays_all = []  # initialize list to store [rays_o, rays_d]
        val_idxs = torch.Tensor([i+len(self.meta_train["frames"]) for i in range(len(self.meta_test["frames"]))])

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times), val_idxs
    def get_render_rays(self):
        render_poses = self.load_render_views()
        # render_poses = torch.stack(render_poses,0)
        render_times = torch.Tensor([i for i in range(len(render_poses))])
        render_times = 2*(render_times - render_times.min())/(render_times.max() - render_times.min()+1e-6)-1
        render_idxs = torch.ones(len(render_poses))
        rays_all = []
        for i in range(len(render_poses)):
            c2w = torch.FloatTensor(render_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)
            rays_all.append(rays)
        assert len(render_poses) == len(render_times) and len(render_times) == len(render_idxs)
        return rays_all, torch.FloatTensor(render_times), render_idxs
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

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """
        Apply Geometry Regularization from RegNeRF.
        This function randomly samples many patches from random poses.
        """
        n_patches = batch_size // (patch_size**2)

        N_random = self.random_rays.shape[0]
        # Sample images
        if batching == "all_images":
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        elif batching == "single_image":
            idx_img = np.random.randint(0, N_random)
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            raise ValueError("Not supported batching type!")
        idx_img = torch.Tensor(idx_img).long()
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        xy0 = np.concatenate([x0, y0], axis=-1)
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        return out, self.random_times[idx_img]
    def load_render_views(self):
        with open(os.path.join(self.root_dir,"transforms_render.json")) as f:
            self.meta_render = json.load(f)
        render_poses = []
        for frame in self.meta_render["frames"]:
            render_poses.append(frame["transform_matrix"])
        return render_poses