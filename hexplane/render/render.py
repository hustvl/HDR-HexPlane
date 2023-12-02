import os

import imageio
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm

from hexplane.render.util.metric import rgb_lpips, rgb_ssim
from hexplane.render.util.util import visualize_depth_numpy
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
tonemap = lambda x: (np.log(np.clip(x, 0, 1) * 5000 + 1) / np.log(5000 + 1) * 255).astype(np.uint8)
import cv2
def OctreeRender_trilinear_fast(
    rays,
    time,
    idx,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
    exps=None,
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals, rgbh = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        idx_chunk = idx[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        if exps is not None:
            # exps_chunk = None
            exps_chunk = exps[chunk_idx * chunk : (chunk_idx + 1) * chunk].unsqueeze(-1).to(device) 
        #    idx_chunk = None
        else:
            exps_chunk = None
        rgb_map, depth_map, alpha_map, z_val_map, rgb_h_map = model(
            rays_chunk,
            time_chunk,
            idx_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
            exps=exps_chunk
        )
        # if not is_train:
        #     rgb_map = rgb_map.to("cpu")
        #     depth_map = depth_map.to("cpu")
        #     alpha_map = alpha_map.to("cpu")
        #     z_val_map = z_val_map.to("cpu")
        #     rgb_h_map = rgb_h_map.to("cpu")
            
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
        rgbh.append(rgb_h_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        torch.cat(rgbh),
        None,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    model.eval()
    PSNRs, rgb_maps, depth_maps, gt_depth_maps, rgbh_maps = [], [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    W, H = test_dataset.W, test_dataset.H
    for idx in tqdm(idxs):
        data = test_dataset[idx]
        samples, gt_rgb, sample_times, sample_idxs = data["rays"], data["rgbs"], data["time"], data["idx"]
        if prefix != "train":
            sample_idxs  = sample_idxs.clone() + torch.ones_like(sample_idxs) * (model.train_images - model.val_images)
        depth = None


        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])
        sample_idx = sample_idxs.view(-1, sample_idxs.shape[-1])
        rgb_map, _, depth_map, _, rgbh_map, _ = OctreeRender_trilinear_fast(
            rays,
            times,
            sample_idx,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        # rgbh_map = rgbh_map.clamp(0,1)
        # rgbh_map = rgbh_map / torch.max(rgbh_map,0).values

        tmp = torch.Tensor()
        tmp1 = torch.Tensor()
        rgb_map, depth_map, rgbh_map = (
            rgb_map.reshape(H[idx], W[idx], 3).cpu(),
            depth_map.reshape(H[idx], W[idx]).cpu(),
            rgbh_map.reshape(H[idx], W[idx], 3).cpu(),
        )




        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depth" in data.keys():
            depth = data["depth"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H[idx], W[idx], 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                try:
                    ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                except:
                    ms_ssim=0
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        # rgbh_map = torch.log(rgbh_map)
        tonema2 =  cv2.createTonemap(gamma=2.2)
        rgbh_map_tone = tonemap(rgbh_map.numpy()/np.max(rgbh_map.numpy()))
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        rgbh_maps.append(rgbh_map_tone)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_tonemap.png", rgbh_map_tone)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.exr", rgbh_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)
    rgbh_maps_final, rgb_maps_final, depth_maps_final = [], [], []
    for idx, (rgb_map, rgbh_map, depth_map) in enumerate(zip(rgb_maps, rgbh_maps, depth_maps)):
        if rgb_map.shape[0] != rgb_map.shape[1]:
            tmp = np.zeros([test_dataset.H[idx], test_dataset.W[idx]])
            tmp1 = np.zeros([test_dataset.H[idx], test_dataset.W[idx], 3])

            rgb_map = np.concatenate([rgb_map, tmp1],1)
            rgbh_map = np.concatenate([rgbh_map, tmp1], 1)
            depth_map = np.concatenate([depth_map, tmp1], 1)
        rgbh_maps_final.append(rgbh_map)
        rgb_maps_final.append(rgb_map)
        depth_maps_final.append(depth_map)
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps_final).astype(np.uint8),
        fps=30,
        format="FFMPEG",
        quality=10,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}_HDR_video.mp4",
        np.stack(rgbh_maps_final).astype(np.uint8),
        fps=30,
        format="FFMPEG",
        quality=10,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps_final).astype(np.uint8),
        format="FFMPEG",
        fps=30,
        quality=10,
    )
    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}_gt_depthvideo.mp4",
            np.stack(gt_depth_maps).astype(np.uint8),
            format="FFMPEG",
            fps=30,
            quality=10,
        )

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")
    model.train()
    return PSNRs


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=10,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    model.eval()
    rgb_maps, depth_maps, rgbh_maps, rgbh_tone = [], [], [] ,[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times, val_idx = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        idxs = val_idx[idx]
        time = time.expand(rays.shape[0], 1)
        idxs = idxs.expand(rays.shape[0], 1)
        # idxs = torch.ones_like(time)*idx + model.train_images - model.val_images
        idxs = torch.tensor(idxs, dtype=torch.int32)
        rgb_map, _, depth_map, _, rgbh_map, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            idxs,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        # rgbh_map = rgbh_map/torch.max(rgbh_map,0).values
        rgb_map, depth_map, rgbh_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
            rgbh_map.reshape(H, W, 3).cpu()
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        
        rgb_maps.append(rgb_map)
        # tonemap =  cv2.createTonemap(gamma=2.2)
        rgbh_map_tone = tonemap(rgbh_map.numpy()/np.max(rgbh_map.numpy()))
        # rgbh_map=(tonemap.process(rgbh_map.cpu().numpy()) * 255).astype(np.uint8)
        rgbh_maps.append(rgbh_map)
        depth_maps.append(depth_map)
        rgbh_tone.append(rgbh_map_tone)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_tonemap.png", rgbh_map_tone)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.exr",rgbh_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            
    imageio.mimwrite(
        f"{savePath}/{prefix}HDRvideo.mp4", np.stack(rgbh_tone), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )
    model.train()
    return 0
@torch.no_grad()
def evaluation_hdr(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=10,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    model.eval()
    rgb_maps_1, rgb_maps_2, depth_maps, rgbh_maps = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times, val_idx = test_dataset.get_val_rays()
    tonemap1 = cv2.createTonemap(gamma=2.2)
    # render_exposure = model.get_render_exposures()
    (min_idx, exp_min), (max_idx, exp_max) = model.get_mutiexposures_index()   
         
    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        idxs = val_idx[idx]
        time = time.expand(rays.shape[0], 1)
        idxs = idxs.expand(rays.shape[0], 1)
        # idxs = torch.ones_like(time)*idx + model.train_images - model.val_images
        idxs = torch.tensor(idxs, dtype=torch.int32)
        idx1 = torch.ones_like(idxs) * min_idx.cpu()
        idx2 = torch.ones_like(idxs) * max_idx.cpu()
        rgb_map1, _, depth_map, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            idx1,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            )
        rgb_map2, _, _, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            idx2,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        img_list = [rgb_map1.reshape(H,W,3).clone().detach().cpu().numpy(), rgb_map2.reshape(H,W,3).clone().detach().cpu().numpy()]
        # img_list = [i for i in img_list]
        print("Done")
        merge_debevec = cv2.createMergeDebevec()
        hdr_debevec = merge_debevec.process(src=[to8b(i) for i in img_list], times=np.array([0.2,1],dtype=np.float32))
        res_debevec = tonemap1.process(hdr_debevec.copy())
        calibrate_debevec = cv2.createCalibrateDebevec()
        response = calibrate_debevec.process([to8b(i) for i in img_list], times=np.array([0.2,1],dtype=np.float32))


        res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
        # rgb_map = rgb_map1
        rgb_map1 = rgb_map1.clamp(0.0, 1.0)
        rgb_map2 = rgb_map2.clamp(0.0, 1.0)
        rgb_map1, rgb_map2, depth_map = (
            rgb_map1.reshape(H, W, 3).cpu(),
            rgb_map2.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
            
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        
        rgb_map1, rgb_map2 = (rgb_map1.numpy() * 255).astype("uint8"), (rgb_map2.numpy() * 255).astype("uint8")
        
        rgb_maps_1.append(rgb_map1)
        rgb_maps_2.append(rgb_map2)
        rgbh_maps.append(res_debevec_8bit)
        depth_maps.append(depth_map)
        if savePath is not None:
            import matplotlib.pyplot as plt
            response_1d = response.ravel()

            plt.plot(response_1d)
            plt.title("Camera Response Function")
            plt.xlabel("Pixel Value")
            plt.ylabel("Response")
            plt.savefig(f"{savePath}/{prefix}{idx:03d}_CRF.png")
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_LDR1.png", rgb_map1)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_LDR2.png", rgb_map2)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_HDR.png", res_debevec_8bit)
            rgb_map = np.concatenate((rgb_map1, rgb_map2, res_debevec_8bit, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
    imageio.mimwrite(
        f"{savePath}/{prefix}video_HDR.mp4", np.stack(rgbh_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}video_LDR1.mp4", np.stack(rgb_maps_1), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}video_LDR2.mp4", np.stack(rgb_maps_2), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )
    model.train()
    return 0
def evaluation_mutiexp(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=10,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    model.eval()
    rgb_maps_1, rgb_maps_2, depth_maps, rgbh_maps = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times, val_idx = test_dataset.get_val_rays()
    # render_exposure = model.get_render_exposures()
    # (min_idx, exp_min), (max_idx, exp_max) = model.get_mutiexposures_index()   
    # def get_render_exposure(idx, max_idx):
    #     if idx <= max_idx //2 :
    #         exp = exp_min + (exp_max - exp_min) * (idx/(max_idx//2))
    #     else:
    #         exp = exp_max - (exp_max - exp_min) * (idx/(max_idx//2))
    #     return exp
    for idx in tqdm(range(val_times.shape[0])):
        # exps = get_render_exposure(idx,val_times.shape[0])
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        idxs = val_idx[idx]
        time = time.expand(rays.shape[0], 1)
        idxs = idxs.expand(rays.shape[0], 1)
        # idxs = torch.ones_like(time)*idx + model.train_images - model.val_images
        idxs = torch.tensor(idxs, dtype=torch.int32)
        # idx1 = torch.ones_like(idxs) * min_idx.cpu()
        # exps = torch.ones_like(idxs) * exps.cpu()
        rgb_map, _, depth_map, _, rgbh_map, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            idxs,
            model,
            chunk=1024,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            exps=None
            )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map1, rgbh_map, depth_map = (
            rgb_map1.reshape(H, W, 3).cpu(),
            rgbh_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        
            
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        
        rgb_map1 = (rgb_map1.numpy() * 255).astype("uint8")
        
        rgb_maps_1.append(rgb_map1)
        depth_maps.append(depth_map)
        if savePath is not None:
            import matplotlib.pyplot as plt
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_LDR1.png", rgb_map1)
            # imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_HDR.png", res_debevec_8bit)
            rgb_map = np.concatenate((rgb_map1, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
    # imageio.mimwrite(
    #     f"{savePath}/{prefix}video_HDR.mp4", np.stack(rgbh_maps), fps=30, quality=8
    # )
    imageio.mimwrite(
        f"{savePath}/{prefix}video_LDR.mp4", np.stack(rgb_maps_1), fps=30, quality=8
    )
    model.train()
    return 0

@torch.no_grad()
def evaluation_video(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=10,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    model.eval()
    rgb_maps, depth_maps, rgbh_maps = [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times, val_idx = test_dataset.get_render_rays()
    render_exposure = model.get_render_exposures()
    (min_idx, exp_min), (max_idx, exp_max) = model.get_mutiexposures_index()   
    def get_render_exposure(idx, max_idx):
        if idx <= max_idx //2 :
            exp = exp_min + (exp_max - exp_min) * (idx/(max_idx//2))
        else:
            exp = exp_max - (exp_max - exp_min) * ((idx-max_idx//2)/(max_idx//2))
        return exp
    for idx in tqdm(range(val_times.shape[0])):
        exps = get_render_exposure(idx,val_times.shape[0])

        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        idxs = val_idx[idx]
        time = time.expand(rays.shape[0], 1)
        idxs = idxs.expand(rays.shape[0], 1)
        # idxs = torch.ones_like(time)*idx + model.train_images - model.val_images
        idxs = torch.tensor(idxs, dtype=torch.int32)
        exps = torch.ones_like(idxs) * exps.cpu()

        rgb_map, _, depth_map, _, rgbh_map, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            idxs,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            exps = exps
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        # rgbh_map = rgbh_map/torch.max(rgbh_map,0).values
        rgb_map, depth_map, rgbh_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
            rgbh_map.reshape(H, W, 3).cpu()
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        
        rgb_maps.append(rgb_map)
        # tonemap =  cv2.createTonemap(gamma=2.2)
        rgbh_map = tonemap(rgbh_map.numpy()/np.max(rgbh_map.numpy()))
        # rgbh_map=(tonemap.process(rgbh_map.cpu().numpy()) * 255).astype(np.uint8)
        rgbh_maps.append(rgbh_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_HDR.png", rgbh_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
    imageio.mimwrite(
        f"{savePath}/{prefix}HDRvideo.mp4", np.stack(rgbh_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )
    model.train()
    return 0