# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import json
import datetime

from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="DCVC Training")
    # Model configuration
    parser.add_argument('--i_frame_model_name', type=str, default="cheng2020-anchor")
    parser.add_argument('--i_frame_model_path', type=str, default=None,
                        help="Path to pretrained I-frame model")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to pretrained model for finetuning")
    parser.add_argument('--model_type', type=str, default="psnr",
                        help="Training objective: psnr or msssim")

    # Training configuration
    parser.add_argument('--train_config', type=str, required=True,
                        help="JSON configuration file for training dataset")
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument('--log_dir', type=str, default="logs",
                        help="Directory to save tensorboard logs")
    parser.add_argument('--save_interval', type=int, default=5000,
                        help="Save model every n steps")
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help="Evaluate model every n steps")

    # Training hyperparameters
    parser.add_argument('--lambda', type=float, default=1024,
                        help="Rate-distortion trade-off parameter")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help="Learning rate decay factor")
    parser.add_argument('--lr_decay_steps', type=int, nargs='+', default=[200000, 400000, 500000],
                        help="Steps at which to decay learning rate")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Training batch size")
    parser.add_argument('--train_steps', type=int, default=600000,
                        help="Total number of training steps")
    parser.add_argument('--gop', type=int, default=10,
                        help="GOP size for training")
    parser.add_argument('--patch_size', type=int, default=256,
                        help="Training patch size")

    # Training stages
    parser.add_argument('--stage', type=int, default=4,
                        help="Training stage (1-4)")
    parser.add_argument('--freeze_me', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Freeze motion estimation module")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume training from checkpoint")

    # Hardware options
    parser.add_argument('--cuda', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--cuda_device', default=None,
                        help="CUDA device(s) to use, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loading workers")

    args = parser.parse_args()
    return args


class VideoPairDataset(torch.utils.data.Dataset):
    """Dataset for loading video frames in pairs (reference, current) for training"""

    def __init__(self, config, patch_size=256, gop=10):
        super().__init__()
        self.config = config
        self.sequences = []
        self.patch_size = patch_size
        self.gop = gop

        # Process training dataset configuration
        for ds_name in config:
            for seq_name in config[ds_name]['sequences']:
                seq_info = config[ds_name]['sequences'][seq_name]
                base_path = config[ds_name]['base_path']
                self.sequences.append({
                    'name': seq_name,
                    'path': os.path.join(base_path, seq_name),
                    'frames': seq_info['frames']
                })

        # Create frame pairs (ref_frame, current_frame) for training
        self.frame_pairs = []

        for seq in self.sequences:
            # Check file naming pattern
            files = os.listdir(seq['path'])
            if 'im1.png' in files:
                padding = 1
            elif 'im00001.png' in files:
                padding = 5
            else:
                raise ValueError(f"Unknown image naming convention in {seq['path']}")

            seq['padding'] = padding

            # Create pairs while respecting GOP structure
            for i in range(1, seq['frames']):
                # Only use P-frames for training
                if i % self.gop != 0:
                    # Find the reference frame (previous I-frame or P-frame)
                    ref_idx = i - 1
                    if ref_idx % self.gop == 0:
                        # Previous frame is an I-frame
                        ref_type = 'i_frame'
                    else:
                        # Previous frame is a P-frame
                        ref_type = 'p_frame'

                    self.frame_pairs.append({
                        'sequence': seq['name'],
                        'path': seq['path'],
                        'padding': padding,
                        'ref_idx': ref_idx,
                        'cur_idx': i,
                        'ref_type': ref_type
                    })

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, index):
        pair = self.frame_pairs[index]

        # Load reference frame
        ref_path = os.path.join(pair['path'], f"im{str(pair['ref_idx'] + 1).zfill(pair['padding'])}.png")
        ref_img = Image.open(ref_path).convert('RGB')

        # Load current frame
        cur_path = os.path.join(pair['path'], f"im{str(pair['cur_idx'] + 1).zfill(pair['padding'])}.png")
        cur_img = Image.open(cur_path).convert('RGB')

        # Random crop both frames at the same position
        width, height = ref_img.size

        # Ensure images are large enough for the patch
        if width < self.patch_size or height < self.patch_size:
            # Scale up if needed
            scale = max(self.patch_size / width, self.patch_size / height) * 1.1
            new_width = int(width * scale)
            new_height = int(height * scale)
            ref_img = ref_img.resize((new_width, new_height), Image.BICUBIC)
            cur_img = cur_img.resize((new_width, new_height), Image.BICUBIC)
            width, height = new_width, new_height

        # Random crop
        x = random.randint(0, width - self.patch_size)
        y = random.randint(0, height - self.patch_size)

        ref_img = ref_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        cur_img = cur_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Convert to torch tensors and normalize to [0, 1]
        ref_tensor = torch.from_numpy(np.array(ref_img).astype('float32').transpose(2, 0, 1)) / 255.0
        cur_tensor = torch.from_numpy(np.array(cur_img).astype('float32').transpose(2, 0, 1)) / 255.0

        # Apply random augmentations (horizontal and vertical flips)
        if random.random() > 0.5:
            ref_tensor = torch.flip(ref_tensor, dims=[2])  # Horizontal flip
            cur_tensor = torch.flip(cur_tensor, dims=[2])

        if random.random() > 0.5:
            ref_tensor = torch.flip(ref_tensor, dims=[1])  # Vertical flip
            cur_tensor = torch.flip(cur_tensor, dims=[1])

        return {
            'ref_frame': ref_tensor,
            'cur_frame': cur_tensor,
            'ref_type': pair['ref_type']
        }


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def train_one_epoch(model, i_frame_model, train_loader, optimizer, device,
                    global_step, args, writer):
    model.train()
    if i_frame_model is not None:
        i_frame_model.eval()

    epoch_loss = 0
    epoch_psnr = 0
    epoch_bpp = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, batch in enumerate(progress_bar):
        ref_frame = batch['ref_frame'].to(device)
        cur_frame = batch['cur_frame'].to(device)
        ref_type = batch['ref_type']

        # Forward pass
        optimizer.zero_grad()
        result = model(ref_frame, cur_frame, training=True, stage=args.stage)

        # Compute loss
        loss = result['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update progress
        epoch_loss += loss.item()
        epoch_psnr += PSNR(result['recon_image'], cur_frame)
        epoch_bpp += result['bpp'].item()

        avg_loss = epoch_loss / (batch_idx + 1)
        avg_psnr = epoch_psnr / (batch_idx + 1)
        avg_bpp = epoch_bpp / (batch_idx + 1)

        # Update progress bar
        progress_bar.set_description(
            f"Step: {global_step} | "
            f"Loss: {avg_loss:.4f} | "
            f"PSNR: {avg_psnr:.2f} dB | "
            f"BPP: {avg_bpp:.4f}"
        )

        # Log to tensorboard
        if global_step % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/psnr', PSNR(result['recon_image'], cur_frame), global_step)
            writer.add_scalar('train/bpp', result['bpp'].item(), global_step)
            writer.add_scalar('train/bpp_mv_y', result['bpp_mv_y'].item(), global_step)
            writer.add_scalar('train/bpp_mv_z', result['bpp_mv_z'].item(), global_step)
            writer.add_scalar('train/bpp_y', result['bpp_y'].item(), global_step)
            writer.add_scalar('train/bpp_z', result['bpp_z'].item(), global_step)

        # Apply learning rate decay
        if global_step in args.lr_decay_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay
                writer.add_scalar('train/lr', param_group['lr'], global_step)
                print(f"Learning rate decayed to {param_group['lr']}")

        # Save checkpoint
        if global_step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"model_step_{global_step}.pth")
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        global_step += 1

        # Stop if reached the maximum number of steps
        if global_step >= args.train_steps:
            break

    return global_step, avg_loss, avg_psnr, avg_bpp


def evaluate(model, i_frame_model, val_loader, device, global_step, args, writer):
    model.eval()
    if i_frame_model is not None:
        i_frame_model.eval()

    total_loss = 0
    total_psnr = 0
    total_bpp = 0
    total_bpp_mv_y = 0
    total_bpp_mv_z = 0
    total_bpp_y = 0
    total_bpp_z = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            ref_frame = batch['ref_frame'].to(device)
            cur_frame = batch['cur_frame'].to(device)

            # Forward pass
            result = model(ref_frame, cur_frame, training=False, stage=args.stage)

            # Compute metrics
            total_loss += result['loss'].item()
            total_psnr += PSNR(result['recon_image'], cur_frame)
            total_bpp += result['bpp'].item()
            total_bpp_mv_y += result['bpp_mv_y'].item()
            total_bpp_mv_z += result['bpp_mv_z'].item()
            total_bpp_y += result['bpp_y'].item()
            total_bpp_z += result['bpp_z'].item()

    # Average metrics
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_bpp = total_bpp / len(val_loader)
    avg_bpp_mv_y = total_bpp_mv_y / len(val_loader)
    avg_bpp_mv_z = total_bpp_mv_z / len(val_loader)
    avg_bpp_y = total_bpp_y / len(val_loader)
    avg_bpp_z = total_bpp_z / len(val_loader)

    # Log to tensorboard
    writer.add_scalar('val/loss', avg_loss, global_step)
    writer.add_scalar('val/psnr', avg_psnr, global_step)
    writer.add_scalar('val/bpp', avg_bpp, global_step)
    writer.add_scalar('val/bpp_mv_y', avg_bpp_mv_y, global_step)
    writer.add_scalar('val/bpp_mv_z', avg_bpp_mv_z, global_step)
    writer.add_scalar('val/bpp_y', avg_bpp_y, global_step)
    writer.add_scalar('val/bpp_z', avg_bpp_z, global_step)

    print(f"\nEvaluation Results at step {global_step}:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"BPP: {avg_bpp:.4f}")

    return avg_loss, avg_psnr, avg_bpp


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up device
    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir, f"{current_time}_stage{args.stage}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Load dataset configuration
    with open(args.train_config, 'r') as f:
        train_config = json.load(f)

    # Create training dataset and dataloader
    train_dataset = VideoPairDataset(train_config, patch_size=args.patch_size, gop=args.gop)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"Loaded {len(train_dataset)} training frame pairs.")

    # Create validation dataset (using the same dataset with a different random seed)
    # This is a simplified approach; ideally we'd have a separate validation set
    val_dataset = VideoPairDataset(train_config, patch_size=args.patch_size, gop=args.gop)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create model
    model = DCVC_net(lmbda=getattr(args, 'lambda'))  # Using getattr since 'lambda' is a Python keyword
    model = model.to(device)

    # Load I-frame model if needed
    i_frame_model = None
    if args.i_frame_model_path is not None:
        i_frame_checkpoint = torch.load(args.i_frame_model_path, map_location=device)
        i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_checkpoint)
        i_frame_model = i_frame_model.to(device)
        i_frame_model.eval()  # Set to evaluation mode since we don't train it

    # Freeze motion estimation module if needed
    if args.freeze_me:
        model.opticFlow.requires_grad_(False)
        print("Motion estimation module frozen.")

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Load pretrained model or resume training if needed
    start_step = 0
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_dict(checkpoint)
        print(f"Loaded pretrained model from {args.model_path}")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resumed training from step {start_step}")

    # Log model architecture
    writer.add_text('model_architecture', str(model))

    # Log hyperparameters
    hparams = {
        'lambda': getattr(args, 'lambda'),
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'gop': args.gop,
        'stage': args.stage,
        'freeze_me': args.freeze_me,
        'model_type': args.model_type
    }
    writer.add_hparams(hparams, {})

    # Training loop
    global_step = start_step

    try:
        while global_step < args.train_steps:
            # Train one epoch
            global_step, avg_loss, avg_psnr, avg_bpp = train_one_epoch(
                model, i_frame_model, train_loader, optimizer, device,
                global_step, args, writer
            )

            # Evaluate model
            if global_step % args.eval_interval == 0:
                eval_loss, eval_psnr, eval_bpp = evaluate(
                    model, i_frame_model, val_loader, device, global_step, args, writer
                )

    except KeyboardInterrupt:
        print("Training interrupted.")

    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, f"model_final_stage{args.stage}.pth")
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    # Close tensorboard writer
    writer.close()

    print("Training completed.")


if __name__ == "__main__":
    main()

# python train_dcvc.py --train_config dataset_config.json --stage 1 --lambda 1024 --output_dir checkpoints/stage1 --log_dir logs/stage1 --cuda true --cuda_device 0,1
# python train_dcvc.py --train_config dataset_config.json --stage 2 --lambda 1024 --output_dir checkpoints/stage2 --log_dir logs/stage2 --freeze_me true --model_path checkpoints/stage1/model_final_stage1.pth --cuda true --cuda_device 0,1
# python train_dcvc.py --train_config dataset_config.json --stage 3 --lambda 1024 --output_dir checkpoints/stage3 --log_dir logs/stage3 --freeze_me true --model_path checkpoints/stage2/model_final_stage2.pth --cuda true --cuda_device 0,1
# python train_dcvc.py --train_config dataset_config.json --stage 4 --lambda 1024 --output_dir checkpoints/stage4 --log_dir logs/stage4 --model_path checkpoints/stage3/model_final_stage3.pth --cuda true --cuda_device 0,1