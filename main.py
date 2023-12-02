import datetime
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
# from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.model import init_model
from hexplane.render.render import evaluation, evaluation_path, evaluation_hdr, evaluation_video, evaluation_mutiexp
from hexplane.render.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_test(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    if not os.path.exists(cfg.systems.ckpt):
        print("the ckpt path does not exists!!")
        return

    HexPlane = torch.load(cfg.systems.ckpt, map_location=device)
    logfolder = os.path.dirname(cfg.systems.ckpt)

    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


def reconstruction(cfg):
    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far

    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    # summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # init model.
    aabb = train_dataset.scene_bbox.to(device)
    train_image = train_dataset.get_total_image()
    test_image = test_dataset.get_total_image()
    video_split_index = train_dataset.get_video_split_index()
    total_image = [train_image+test_image, train_image, test_image]
    print("total_image:",total_image)
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device, total_image, video_split_index)

    # init trainer.
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        None,
        logfolder,
        device,
    )
    if cfg.systems.ckpt is None:
        trainer.train()
        torch.save(HexPlane, f"{logfolder}/{cfg.expname}.th")
    # Render training viewpoints.
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render test viewpoints.
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render validation viewpoints.
    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    if cfg.render_hdr:
        os.makedirs(f"{logfolder}/imgs_hdr_all", exist_ok=True)
        evaluation_hdr(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_hdr_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    if cfg.render_video:
        os.makedirs(f"{logfolder}/imgs_video", exist_ok=True)
        evaluation_video(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_video/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    if cfg.render_mutiexp:
        os.makedirs(f"{logfolder}/imgs_video", exist_ok=True)
        evaluation_mutiexp(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_mutiexp/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    if cfg.draw_exp:
        import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.hist(HexPlane.get_render_exposures().detach().cpu().numpy(), bins=50)

        plt.xlabel('Exposure',fontsize=18)
        plt.ylabel('Images',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{logfolder}/{cfg.expname}.pdf",format='pdf')
        plt.savefig(f"{logfolder}/{cfg.expname}.png",format='png')
if __name__ == "__main__":
    # Load config file from base config, yaml and cli.

    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
