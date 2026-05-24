import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, prefilter_voxel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loader_utils import camera_name_parts
from utils.loss_utils import l1_loss, ssim

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
    print("3DGS-only edit: frozen MLP/appearance groups; anchor growing/pruning disabled.")
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


def camera_stem(cam):
    return Path(str(getattr(cam, "image_name", getattr(cam, "uid", "camera")))).stem


def find_target_image(edited_dir, cam, index, pattern):
    edited_dir = Path(edited_dir)
    stem, camera_id, frame_id = camera_name_parts(cam)
    if pattern:
        positional_id = camera_id if camera_id is not None else index
        values = {
            "image_name": stem,
            "uid": getattr(cam, "uid", index),
            "index": index,
            "camera_id": camera_id if camera_id is not None else index,
            "frame_id": frame_id if frame_id is not None else index,
            "timestamp": getattr(cam, "timestamp", 0.0),
        }
        try:
            filename = pattern.format(positional_id, **values)
        except (IndexError, KeyError):
            filename = pattern.format(**values)
        path = edited_dir / filename
        if path.exists():
            return path

    for ext in (".png", ".jpg", ".jpeg"):
        names = [stem, str(index), f"{index:05d}", str(getattr(cam, "uid", index))]
        if camera_id is not None:
            names.extend([str(camera_id), f"{camera_id:02d}", f"{camera_id:05d}"])
        for name in names:
            path = edited_dir / f"{name}{ext}"
            if path.exists():
                return path
    return None


def build_target_pairs(train_views, args):
    pairs = []
    missing = 0
    for idx, (gt, cam) in enumerate(train_views):
        target_path = find_target_image(args.edited_images_path, cam, idx, args.edited_pattern)
        if target_path is None:
            missing += 1
            # if args.fallback_original:
            #     pairs.append((idx, gt, cam, None))
        else:
            pairs.append((idx, gt, cam, target_path))

    if not pairs:
        raise RuntimeError(
            f"No edited targets were matched in {args.edited_images_path}. "
            "Use --edited_pattern like 'edited_statue_original_time0_{}.png' "
            "or 'edited_statue_original_time0_{index}.png'."
        )

    if missing:
        print(f"Matched {len(pairs)} training views with targets; skipped {missing} views without edited images.")
    else:
        print(f"Matched all {len(pairs)} training views with edited targets.")
    return pairs


def load_target(path, size):
    image = Image.open(path).convert("RGB")
    target = torchvision.transforms.functional.to_tensor(image).cuda()
    target = F.interpolate(target.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return target.clamp(0, 1)


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


def report(tb_writer, iteration, scene, gaussians, pipe, background, loss_value, l1_value):
    if tb_writer is None:
        return
    tb_writer.add_scalar("edit/train_loss/l1", l1_value.item(), iteration)
    tb_writer.add_scalar("edit/train_loss/total", loss_value.item(), iteration)
    tb_writer.add_scalar("edit/total_anchors", gaussians.get_anchor.shape[0], iteration)

    with torch.no_grad():
        views = list(scene.getTestCameras())
        if not views:
            return
        gt, cam = views[min(iteration % len(views), len(views) - 1)]
        image = render_view(cam, gaussians, pipe, background, retain_grad=False)["render"].clamp(0, 1)
        tb_writer.add_images("edit/test/render", image[None], iteration)
        tb_writer.add_scalar("edit/test/psnr_to_original", psnr(image, gt.cuda()).mean().item(), iteration)


def train_edit(dataset, opt, pipe, args):
    tb_writer = prepare_output(args)
    dataset.dataloader = True
    dataset.skip_test_cameras = not args.include_test_targets
    dataset.checkpoint_dir = args.checkpoint_dir
    dataset.mlp_dir = args.mlp_dir
    gaussians = make_gaussians(dataset)
    direct_model = args.checkpoint_dir or (args.ply_path and args.mlp_dir)
    load_iteration = None if direct_model and args.iteration == -1 else args.iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    configure_3dgs_only_training(gaussians, args)
    gaussians.train()

    background = background_tensor(dataset)
    train_views = list(scene.getTrainCameras(view_only=not args.fallback_original))
    if args.include_test_targets:
        train_views.extend(list(scene.getTestCameras(view_only=not args.fallback_original)))
    if not train_views:
        raise RuntimeError("No training cameras were loaded.")

    target_pairs = build_target_pairs(train_views, args)
    progress = tqdm(range(1, opt.iterations + 1), desc="3D edit")
    ema = 0.0
    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        freeze_optimizer_groups(gaussians, ("mlp_", "embedding_appearance"))
        batch = random.sample(target_pairs, min(args.batch_size, len(target_pairs)))

        images = []
        targets = []
        render_pkgs = []
        stat_cams = []
        for idx, gt, cam, target_path in batch:
            retain_grad = args.anchor_update and iteration < opt.update_until
            pkg = render_view(cam, gaussians, pipe, background, retain_grad=retain_grad)
            image = pkg["render"]
            if target_path is None:
                target = gt.cuda()
            else:
                target = load_target(target_path, image.shape[-2:])
            images.append(image.unsqueeze(0))
            targets.append(target[:3].unsqueeze(0))
            if retain_grad:
                render_pkgs.append(pkg)
                stat_cams.append(cam)

        image_tensor = torch.cat(images, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        l1 = l1_loss(image_tensor, target_tensor)
        loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim(image_tensor, target_tensor))
        loss.backward()

        if torch.isnan(loss):
            raise RuntimeError("Loss became NaN during editing.")

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

        if iteration in args.save_iterations:
            print(f"\\n[ITER {iteration}] Saving edited 4D-Scaffold-GS")
            scene.save_3dedit(iteration, args.prompt)
        if iteration in args.test_iterations:
            report(tb_writer, iteration, scene, gaussians, pipe, background, loss, l1)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Supervised edit/refit for a 4D-Scaffold-GS model saved by train.py")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Source train.py iteration to load from point_cloud/iteration_*")
    parser.add_argument("--checkpoint_dir", default="", type=str, help="Directory containing point_cloud.ply and *_mlp.pt files.")
    parser.add_argument("--mlp_dir", default="", type=str, help="Directory containing opacity/cov/color/flow MLP .pt files. Use with --ply_path.")
    parser.add_argument("--edited_images_path", default="", type=str, help="Directory containing edited RGB targets.")
    parser.add_argument("--edited_pattern", default="", type=str, help="Optional pattern, e.g. '{image_name}.png' or '{index:05d}.png'.")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--freeze_mlp", action="store_true", default=True, help="Deprecated: MLPs are always frozen in 3DGS-only editing.")
    parser.add_argument("--anchor_update", dest="anchor_update", action="store_true", default=False, help="Deprecated: anchor growing/pruning is disabled in 3DGS-only editing.")
    parser.add_argument("--disable_anchor_update", dest="anchor_update", action="store_false", help="Keep anchor count fixed.")
    parser.add_argument("--include_test_targets", action="store_true", default=False, help="Also use test camera metadata when matching edited targets.")
    parser.add_argument("--train_only_targets", dest="include_test_targets", action="store_false", help="Use train camera metadata only.")
    parser.add_argument("--fallback_original", action="store_true", help="Use original training images when an edited target is missing.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--dataset", default="", type=str, help="Kept for compatibility with old launch scripts.")
    parser.add_argument("--scene", default="", type=str, help="Kept for compatibility with old launch scripts.")
    parser.add_argument("--prompt", default="", type=str, help="Kept for compatibility with old launch scripts.")
    args = get_combined_args(parser)

    if not args.edited_images_path:
        if args.dataset and args.scene and args.prompt:
            args.edited_images_path = f"./data/{args.dataset}/{args.scene}/{args.prompt.split(' ')[-1].replace('?', '')}"
        else:
            print("Please pass --edited_images_path.", file=sys.stderr)
            sys.exit(2)
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))

    print("Editing", args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_edit(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\\nEditing complete.")
