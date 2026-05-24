import math
import os
import random
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.loader_utils import FineSampler

from pytorch_lightning import seed_everything   

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


def freeze_optimizer_groups(gaussians, prefixes):
    frozen = []
    for group in gaussians.optimizer.param_groups:
        name = group.get("name", "")
        if any(name.startswith(prefix) for prefix in prefixes):
            group["lr"] = 0.0
            frozen.append(name)
            for param in group["params"]:
                param.requires_grad_(False)
    return frozen


def configure_3dgs_only_training(gaussians, args):
    args.anchor_update = False
    frozen = freeze_optimizer_groups(gaussians, ("mlp_", "embedding_appearance"))
    print("3DGS-only SDS refine: frozen MLP/appearance groups; anchor growing/pruning disabled.")
    if frozen:
        print("Frozen optimizer groups:", ", ".join(frozen))


def zero_temporal_gradients(gaussians):
    if gaussians._anchor.grad is not None and gaussians._anchor.grad.shape[-1] > 3:
        gaussians._anchor.grad[:, 3] = 0
    if gaussians._offset.grad is not None and gaussians._offset.grad.shape[-1] > 3:
        gaussians._offset.grad[:, :, 3] = 0
    if gaussians._scaling.grad is not None and gaussians._scaling.grad.shape[-1] > 6:
        gaussians._scaling.grad[:, 6:] = 0


def render_view(cam, gaussians, pipe, background, retain_grad=True):
    cam = cam.cuda()
    visible = prefilter_voxel(cam, gaussians, pipe, background)
    render_pkg = render(cam, gaussians, pipe, background, visible_mask=visible, retain_grad=retain_grad)
    render_pkg["anchor_visible_mask"] = visible
    return render_pkg


def update_anchors_from_render_stats(gaussians, render_pkgs, cams, iteration, opt):
    if iteration < opt.update_until and iteration > opt.start_stat:
        activate_da = iteration >= opt.da_start_iter
        for render_pkg, cam in zip(render_pkgs, cams):
            gaussians.training_statis(
                render_pkg["viewspace_points"],
                render_pkg["neural_opacity"],
                render_pkg["visibility_filter"],
                render_pkg["selection_mask"],
                render_pkg["anchor_visible_mask"],
                render_pkg["opacity_t"],
                render_pkg["sigma"],
                opt.lambda_temporal_sigma,
                cam.timestamp,
                opt,
                activate_da,
            )

        if iteration > opt.update_from and iteration % opt.update_interval == 0:
            gaussians.adjust_anchor(
                opt,
                check_interval=opt.update_interval,
                success_threshold=opt.success_threshold,
                grad_threshold=opt.densify_grad_threshold,
                min_opacity=opt.min_opacity,
            )
    elif iteration == opt.update_until:
        for attr in ("opacity_accum", "offset_gradient_accum", "offset_time_accum", "offset_denom", "offset_time_denom"):
            if hasattr(gaussians, attr):
                delattr(gaussians, attr)
        torch.cuda.empty_cache()


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
    seed_everything(20211202)
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


def build_view_loader(train_views, args):
    if args.custom_sampler:
        sampler = FineSampler(
            train_views,
            repeats_per_frame=args.sampler_repeats_per_frame,
            history_mix=args.sampler_history_mix,
        )
        return DataLoader(
            train_views,
            batch_size=args.sequence_length,
            sampler=sampler,
            num_workers=args.sampler_workers,
            collate_fn=list,
            drop_last=True,
        )

    return DataLoader(
        train_views,
        batch_size=args.sequence_length,
        shuffle=True,
        num_workers=args.sampler_workers,
        collate_fn=list,
        drop_last=True,
    )


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
    configure_3dgs_only_training(gaussians, args)
    gaussians.train()

    device = torch.device("cuda:0")
    dtype = torch.float16
    ip2p = build_ip2p(device, dtype)
    background = background_tensor(dataset)
    train_views = scene.getTrainCameras()
    if len(train_views) < args.sequence_length:
        raise RuntimeError(f"Need at least {args.sequence_length} training views for SDS refinement.")
    view_loader = build_view_loader(train_views, args)
    view_iter = iter(view_loader)

    progress = tqdm(range(1, opt.iterations + 1), desc="SDS refine")
    ema = 0.0
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        freeze_optimizer_groups(gaussians, ("mlp_", "embedding_appearance"))
        try:
            batch = next(view_iter)
        except StopIteration:
            view_iter = iter(view_loader)
            batch = next(view_iter)

        renders = []
        conds = []
        render_pkgs = []
        stat_cams = []
        for gt, cam in batch:
            retain_grad = args.anchor_update and iteration < opt.update_until
            pkg = render_view(cam, gaussians, pipe, background, retain_grad=retain_grad)
            renders.append(pkg["render"].unsqueeze(0))
            conds.append(gt[:3].cuda().unsqueeze(0))
            if retain_grad:
                render_pkgs.append(pkg)
                stat_cams.append(cam)
        render_tensor = torch.cat(renders, dim=0).clamp(0, 1)
        cond_tensor = torch.cat(conds, dim=0).clamp(0, 1)

        loss = sds_loss(ip2p, render_tensor, cond_tensor, args.prompt, args, device, dtype)
        loss.backward()
        if torch.isnan(loss):
            raise RuntimeError("SDS loss became NaN during refinement.")

        zero_temporal_gradients(gaussians)

        if args.anchor_update:
            with torch.no_grad():
                update_anchors_from_render_stats(gaussians, render_pkgs, stat_cams, iteration, opt)

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
    parser.add_argument("--custom_sampler", action="store_true", help="Use utils.loader_utils.FineSampler for frame-aware view sampling.")
    parser.add_argument("--sampler_workers", default=0, type=int)
    parser.add_argument("--sampler_repeats_per_frame", default=4, type=int)
    parser.add_argument("--sampler_history_mix", default=2, type=int)
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--diffusion_steps", default=20, type=int)
    parser.add_argument("--num_train_timesteps", default=1000, type=int)
    parser.add_argument("--t_min", default=0.02, type=float)
    parser.add_argument("--t_max", default=0.98, type=float)
    parser.add_argument("--guidance_scale", default=10.5, type=float)
    parser.add_argument("--image_guidance_scale", default=1.2, type=float)
    parser.add_argument("--freeze_mlp", action="store_true", default=True, help="Deprecated: MLPs are always frozen in 3DGS-only SDS refinement.")
    parser.add_argument("--anchor_update", dest="anchor_update", action="store_true", default=False, help="Deprecated: anchor growing/pruning is disabled in 3DGS-only SDS refinement.")
    parser.add_argument("--disable_anchor_update", dest="anchor_update", action="store_false", help="Keep anchor count fixed.")
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
