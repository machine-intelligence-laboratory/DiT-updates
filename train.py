# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import autoroot
import autorootcwd
import sys
sys.path.append("sbervae")

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path
import argparse
import logging
import os
import shutil
import yaml

from tqdm import tqdm

from models import DiT_models
from diffusion import create_diffusion, create_rfm
from diffusers.models import AutoencoderKL

from dit_updates.vae.adapters.base import VAEAdapter
from dit_updates.vae.adapters.wan_official import WANOfficialAdapter
from dit_updates.data.latent_datasets import LatentsShardDataset
from dit_updates.data.transforms import DiTCenterCrop
from dit_updates.utils.files import resolve_path
from dit_updates.vae.adapters.registry import resolve_adapter


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def _checkpoint_step_from_name(ckpt_path):
    """Best-effort parse of the training step from a checkpoint filename (e.g. '0010000.pt')."""
    try:
        return int(Path(ckpt_path).stem)
    except ValueError:
        return -1


def find_latest_checkpoint(checkpoint_dir):
    """Return the path of the most relevant checkpoint in a directory.

    Prefers an overwritable 'last.pt'; otherwise the highest-numbered '*.pt'
    file. Returns None if no checkpoint is available.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    last_path = os.path.join(checkpoint_dir, "last.pt")
    if os.path.isfile(last_path):
        return last_path
    ckpts = [c for c in glob(os.path.join(checkpoint_dir, "*.pt"))
             if Path(c).stem != "last"]
    if not ckpts:
        return None
    return max(ckpts, key=_checkpoint_step_from_name)


def load_checkpoint(model, ema, opt, ckpt_path, device, logger):
    """Load model/ema/opt state from a checkpoint.

    Returns (train_steps, epoch) to resume from. `epoch` is the epoch index that
    was in progress when the checkpoint was written, so the loop restarts there.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}")
    # `device` may be a bare int (cuda index); torch.load expects a torch.device/string/callable.
    map_location = device if isinstance(device, torch.device) else torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.module.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    opt.load_state_dict(checkpoint["opt"])
    train_steps = checkpoint.get("step", None)
    if train_steps is None:
        train_steps = _checkpoint_step_from_name(ckpt_path)
        if train_steps < 0:
            train_steps = 0
            logger.info("Loaded checkpoint, but couldn't determine step count; resuming from step 0.")
        else:
            logger.info(f"Loaded checkpoint. Resuming from step {train_steps}.")
    else:
        logger.info(f"Loaded checkpoint. Resuming from step {train_steps}.")
    epoch = checkpoint.get("epoch", 0)
    if epoch is None:
        epoch = 0
    logger.info(f"Resuming from epoch {epoch}.")
    return int(train_steps), int(epoch)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def generate_samples(ema_model, vae: VAEAdapter, diffusion, latent_size, device,
                     num_classes, cfg_scale=4.0, num_samples=10, seed=0,
                     objective="ddpm"):
    """
    Generate a grid of sample images from the EMA model for TensorBoard logging.
    Uses classifier-free guidance and a fixed seed for consistent comparison
    across epochs.
    """
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)

    n = num_samples
    z = torch.randn(n, vae.n_channels, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (n,), device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    if objective == "rfm":
        samples = diffusion.p_sample_loop(
            ema_model.forward_with_cfg_flow, z, model_kwargs=model_kwargs
        )
    else:
        samples = diffusion.p_sample_loop(
            ema_model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
    samples, _ = samples.chunk(2, dim=0)

    preprocessor = vae.create_preprocessor()
    samples, _ = vae.decode(samples, denormalize=True)

    torch.random.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)

    samples = preprocessor.inverse(samples).clamp(0, 1)
    grid = make_grid(samples, nrow=5, padding=2)
    return grid


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup the experiment folder (results_dir is used directly; no indexed subfolder):
    resume_ckpt_path = None
    if rank == 0:
        results_dir = str(resolve_path(args.results_dir, "experiment"))
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_dir = f"{results_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

        if getattr(args, "auto_resume", True):
            # By default, resume from the latest checkpoint in this folder (preferring 'last.pt').
            resume_ckpt_path = find_latest_checkpoint(checkpoint_dir)

        shutil.copy2(args.config_path, f"{results_dir}/config.yaml")
        logger = create_logger(results_dir)
        logger.info(f"Experiment directory: {results_dir}")
        if resume_ckpt_path:
            logger.info(f"Resuming training from checkpoint: {resume_ckpt_path}")
        else:
            logger.info("Starting training from scratch.")
    else:
        logger = create_logger(None)

    # Broadcast the resume checkpoint path from rank 0 to all ranks so every
    # process loads the same weights when auto-resuming.
    obj = [resume_ckpt_path]
    dist.broadcast_object_list(obj, src=0)
    resume_ckpt_path = obj[0]

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    objective = getattr(args, "objective", "ddpm")

    # TODO: Remove hardcoded VAE
    vae = resolve_adapter(args.vae, 
                          device=device,
                          latent_norm_type="scale",
                          latent_stats=Path(args.data_path) / "metadata.json")

    # TODO: Make this factor configurable
    latent_size = args.image_size // 8
    learn_sigma = (objective != "rfm")
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=vae.n_channels,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])

    if objective == "rfm":
        rfm_cfg = getattr(args, "rfm", {})
        diffusion = create_rfm(**rfm_cfg)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # ==========================================
    # CHECKPOINT LOADING LOGIC
    # ==========================================
    train_steps = 0
    start_epoch = 0
    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        train_steps, start_epoch = load_checkpoint(model, ema, opt, resume_ckpt_path, device, logger)

    # Setup TensorBoard and sampling diffusion (rank 0 only):
    writer = None
    if rank == 0:
        tb_dir = f"{results_dir}/tensorboard"
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard logs at {tb_dir}")
    if objective == "rfm":
        sample_diffusion = diffusion
    else:
        sample_diffusion = create_diffusion(str(args.num_sampling_steps))

    # Setup data:
    transform = transforms.Compose([
        DiTCenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        vae.create_preprocessor()
    ])

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = LatentsShardDataset(args.data_path,
                                  split="train",
                                  latent_normalizer=vae.latent_normalizer.numpy(),
                                  sample=True,
                                  in_memory=args.data_in_memory)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    # Only force-sync the EMA if we are NOT resuming from a checkpoint
    if not resume_ckpt_path:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    # train_steps = 0  <-- Removed, defined above during checkpoint loading
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        batch_iter = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for x, y in batch_iter:
            x = x.to(device)
            y = y.to(device)
            # with torch.no_grad():
            # Already done by dataset
            # Map input images to latent space + normalize latents:
            # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, train_steps)
                    writer.add_scalar("train/steps_per_sec", steps_per_sec, train_steps)
                    writer.add_scalar("train/epoch", epoch, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # Save an overwritable "last" checkpoint for cheap resume:
            last_ckpt_every = getattr(args, "last_ckpt_every", None) or args.ckpt_every
            if last_ckpt_every > 0 and train_steps % last_ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/last.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved last checkpoint to {checkpoint_path} (step {train_steps})")
                dist.barrier()

        # Generate and log sample images every sample_every epochs:
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            if rank == 0:
                logger.info(f"Generating sample images at epoch {epoch}...")
                grid = generate_samples(
                    ema, vae, sample_diffusion, latent_size, device,
                    num_classes=args.num_classes, cfg_scale=args.cfg_scale,
                    num_samples=10, seed=args.global_seed,
                    objective=objective,
                )
                writer.add_image("samples/ema_generations", grid, epoch)
                writer.flush()
                samples_dir = f"{results_dir}/samples"
                os.makedirs(samples_dir, exist_ok=True)
                grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
                Image.fromarray(grid_np).save(f"{samples_dir}/epoch_{epoch:09d}.jpg")
                logger.info(f"Logged sample grid to TensorBoard and saved to disk (epoch {epoch})")
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if writer is not None:
        writer.close()
    logger.info("Done!")
    cleanup()


def load_config(path: str) -> argparse.Namespace:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = ["data_path", "vae"]
    for key in required:
        if cfg.get(key) is None:
            raise ValueError(f"Required config field '{key}' is missing or null in {path}")

    assert cfg["model"] in DiT_models, f"Unknown model '{cfg['model']}'. Choose from {list(DiT_models.keys())}"
    assert cfg["image_size"] in (256, 512), f"image_size must be 256 or 512, got {cfg['image_size']}"

    cfg.setdefault("objective", "ddpm")
    cfg.setdefault("data_in_memory", False)
    cfg.setdefault("auto_resume", True)
    cfg.setdefault("last_ckpt_every", cfg.get("ckpt_every", 25000))
    assert cfg["objective"] in ("ddpm", "rfm"), f"objective must be 'ddpm' or 'rfm', got {cfg['objective']}"
    if cfg["objective"] == "rfm":
        assert "rfm" in cfg, "Config section 'rfm' is required when objective is 'rfm'"

    return argparse.Namespace(**cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--local-rank", type=int, default=0) # Dummy arg for Sber server compatibility
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="Disable automatic resume and start training from scratch.")

    cli = parser.parse_args()
    args = load_config(cli.config)

    if cli.no_auto_resume:
        args.auto_resume = False
    args.config_path = cli.config

    main(args)
