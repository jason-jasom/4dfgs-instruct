import math
import os
import random
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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


def render_view(cam, gaussians, pipe, background, retain_grad=True):
    cam = cam.cuda()
    visible = prefilter_voxel(cam, gaussians, pipe, background)
    return render(cam, gaussians, pipe, background, visible_mask=visible, retain_grad=retain_grad)


def prepare_output(args):
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_file:
        cfg_file.write(str(Namespace(**vars(args))))
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(args.model_path)
    except Exception:
        return None


def encode_sample(ip2p, images):
    return ip2p.vae.encode(2 * images - 1).latent_dist.sample() * 0.18215


def encode_mode(ip2p, images):
    return ip2p.vae.encode(2 * images - 1).latent_dist.mode()


def build_ip2p(device, dtype):
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
    from ip2p_models.models.ip2p_unet import UNet3DConditionModel

    ddim_source = "CompVis/stable-diffusion-v1-4"
    ip2p_source = "timbrooks/instruct-pix2pix"

    tokenizer = CLIPTokenizer.from_pretrained(ip2p_source, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ip2p_source, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(ip2p_source, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(ip2p_source, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    return InstructPix2PixPipeline(
        vae=vae.to(device, dtype=dtype),
        text_encoder=text_encoder.to(device, dtype=dtype),
        tokenizer=tokenizer,
        unet=unet.to(device, dtype=dtype),
        scheduler=DDIMScheduler.from_pretrained(ddim_source, subfolder="scheduler"),
    )


def resize_for_diffusion(images, max_side):
    _, _, height, width = images.shape
    factor = max_side / max(width, height)
    short = max(64, math.ceil(min(width, height) * factor / 64) * 64)
    factor = short / min(width, height)
    new_width = max(64, int(width * factor) // 64 * 64)
    new_height = max(64, int(height * factor) // 64 * 64)
    return F.interpolate(images, size=(new_height, new_width), mode="bilinear", align_corners=False)


def sds_loss(ip2p, rendered, cond_images, prompt, args, device, dtype):
    sequence_length = rendered.shape[0]
    rendered = resize_for_diffusion(rendered, args.resize).to(device=device, dtype=dtype)
    cond_images = resize_for_diffusion(cond_images, args.resize).to(device=device, dtype=dtype)

    latents = encode_sample(ip2p, rendered)
    image_latents = encode_mode(ip2p, cond_images)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1, f=sequence_length)
    image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", b=1, f=sequence_length)
    uncond_image_latents = torch.zeros_like(image_latents)

    prompt_embeds = ip2p._encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    ip2p.scheduler.config.num_train_timesteps = args.num_train_timesteps
    ip2p.scheduler.set_timesteps(args.diffusion_steps)

    noise = torch.randn_like(latents)
    t = torch.randint(
        int(args.num_train_timesteps * args.t_min),
        int(args.num_train_timesteps * args.t_max),
        [1],
        dtype=torch.long,
        device=device,
    )
    noisy_latents = ip2p.scheduler.add_noise(latents, noise, t)

    image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    latent_model_input = torch.cat([noisy_latents] * 3, dim=0)
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)

    with torch.no_grad():
        noise_pred = ip2p.unet(latent_model_input, t, prompt_embeds, None, None, False)[0]
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + args.guidance_scale * (noise_pred_text - noise_pred_image)
            + args.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )

    alphas = ip2p.scheduler.alphas_cumprod.to(device)
    weight = (1 - alphas[t]).view(-1, 1, 1, 1, 1)
    grad = torch.nan_to_num(weight * (noise_pred - noise))
    target = (noisy_latents - grad).detach()
    return 0.5 * F.mse_loss(noisy_latents.float(), target.float(), reduction="sum") / sequence_length


def train_sds(dataset, opt, pipe, args):
    tb_writer = prepare_output(args)
    dataset.dataloader = True
    dataset.skip_test_cameras = True
    dataset.checkpoint_dir = args.checkpoint_dir
    dataset.mlp_dir = args.mlp_dir
    gaussians = make_gaussians(dataset)
    direct_model = args.checkpoint_dir or (args.ply_path and args.mlp_dir)
    load_iteration = None if direct_model and args.iteration == -1 else args.iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    gaussians.train()

    device = torch.device("cuda:0")
    dtype = torch.float16
    ip2p = build_ip2p(device, dtype)
    background = background_tensor(dataset)
    train_views = scene.getTrainCameras()
    if len(train_views) < args.sequence_length:
        raise RuntimeError(f"Need at least {args.sequence_length} training views for SDS refinement.")

    progress = tqdm(range(1, opt.iterations + 1), desc="SDS refine")
    ema = 0.0
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        batch_indices = random.sample(range(len(train_views)), args.sequence_length)

        renders = []
        conds = []
        for view_idx in batch_indices:
            gt, cam = train_views[view_idx]
            renders.append(render_view(cam, gaussians, pipe, background, retain_grad=True)["render"].unsqueeze(0))
            conds.append(gt[:3].cuda().unsqueeze(0))
        render_tensor = torch.cat(renders, dim=0).clamp(0, 1)
        cond_tensor = torch.cat(conds, dim=0).clamp(0, 1)

        loss = sds_loss(ip2p, render_tensor, cond_tensor, args.prompt, args, device, dtype)
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("SDS loss became NaN during refinement.")

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        ema = 0.4 * loss.item() + 0.6 * ema
        if iteration % 10 == 0:
            progress.set_postfix({"loss": f"{ema:.6f}", "anchors": gaussians.get_anchor.shape[0]})
        if tb_writer is not None and iteration % args.log_interval == 0:
            tb_writer.add_scalar("sds/train_loss", loss.item(), iteration)
            tb_writer.add_scalar("sds/total_anchors", gaussians.get_anchor.shape[0], iteration)
            tb_writer.add_images("sds/train/render", render_tensor[:1], iteration)
        if iteration in args.save_iterations:
            print(f"\\n[ITER {iteration}] Saving SDS-refined 4D-Scaffold-GS")
            scene.save_refine(iteration, args.prompt)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="InstructPix2Pix SDS refinement for a 4D-Scaffold-GS model saved by train.py")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Source train.py iteration to load from point_cloud/iteration_*")
    parser.add_argument("--checkpoint_dir", default="", type=str, help="Directory containing point_cloud.ply and *_mlp.pt files.")
    parser.add_argument("--mlp_dir", default="", type=str, help="Directory containing opacity/cov/color/flow MLP .pt files. Use with --ply_path.")
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--sequence_length", default=4, type=int)
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--diffusion_steps", default=20, type=int)
    parser.add_argument("--num_train_timesteps", default=1000, type=int)
    parser.add_argument("--t_min", default=0.02, type=float)
    parser.add_argument("--t_max", default=0.98, type=float)
    parser.add_argument("--guidance_scale", default=10.5, type=float)
    parser.add_argument("--image_guidance_scale", default=1.2, type=float)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 300, 500, 800])
    parser.add_argument("--log_interval", default=25, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    args = get_combined_args(parser)

    if not args.prompt:
        raise ValueError("Please provide --prompt for SDS refinement.")
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))

    print("SDS refining", args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_sds(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\\nSDS refinement complete.")
