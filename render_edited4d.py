import os
import time
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state


def make_gaussians(dataset):
    return GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size,
        dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
        dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
        dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight,
        dataset.disable_denom_weight, dataset.hparam_beta, dataset.max_init_t,
    )


def background_tensor(dataset):
    return torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")


def render_view(cam, gaussians, pipe, background, color_mode="rgb"):
    cam = cam.cuda()
    visible = prefilter_voxel(cam, gaussians, pipe, background)
    image = render(cam, gaussians, pipe, background, visible_mask=visible, color_mode=color_mode)["render"]
    return image.clamp(0, 1)


def render_split(model_path, name, iteration, views, gaussians, pipe, background, fps, color_mode, output_name, video_name):
    if not views:
        print(f"No {name} cameras to render.")
        return
    run_name = output_name or f"edited_{iteration}"
    out_dir = os.path.join(model_path, name, run_name)
    render_dir = os.path.join(out_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    if not video_name.endswith(".mp4"):
        video_name = f"{video_name}.mp4"
    video_path = os.path.join(out_dir, video_name)
    frames = []

    times = []
    for idx, (_, cam) in enumerate(tqdm(views, desc=f"Rendering {name}")):
        torch.cuda.synchronize()
        start = time.time()
        image = render_view(cam, gaussians, pipe, background, color_mode=color_mode)
        torch.cuda.synchronize()
        times.append(time.time() - start)

        torchvision.utils.save_image(image, os.path.join(render_dir, f"{idx:05d}.png"))
        frame = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimwrite(video_path, frames, fps=fps, macro_block_size=1)
    if len(times) > 5:
        print(f"{name} FPS: {1.0 / np.mean(times[5:]):.4f}")
    print(f"Saved {video_path}")


def render_sets(dataset, iteration, pipe, skip_train, skip_test, fps, color_mode, output_name, video_name):
    with torch.no_grad():
        gaussians = make_gaussians(dataset)
        direct_model = getattr(dataset, "checkpoint_dir", "") or (getattr(dataset, "ply_path", "") and getattr(dataset, "mlp_dir", ""))
        load_iteration = None if direct_model and iteration == -1 else iteration
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
        gaussians.eval()
        background = background_tensor(dataset)

        if not skip_train:
            render_split(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, background, fps, color_mode, output_name, video_name)
        if not skip_test:
            render_split(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, background, fps, color_mode, output_name, video_name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a 4D-Scaffold-GS model saved by train.py or the edit/refine scripts")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--checkpoint_dir", default="", type=str, help="Directory containing point_cloud.ply and *_mlp.pt files.")
    parser.add_argument("--mlp_dir", default="", type=str, help="Directory containing opacity/cov/color/flow MLP .pt files. Use with --ply_path.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--color_mode", default="rgb", choices=["rgb", "flow", "sigma", "mask"])
    parser.add_argument("--output_name", default="", type=str, help="Output subdirectory name under train/test. Default: edited_<iteration>.")
    parser.add_argument("--video_name", default="renders.mp4", type=str, help="Output mp4 filename. .mp4 is appended if omitted.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering", args.model_path)
    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.checkpoint_dir = args.checkpoint_dir
    dataset.mlp_dir = args.mlp_dir
    render_sets(dataset, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.fps, args.color_mode, args.output_name, args.video_name)
