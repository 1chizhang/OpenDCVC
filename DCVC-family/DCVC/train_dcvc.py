# train_dcvc.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
import torchvision.transforms as transforms
import time

# Import the model
from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures


# Define a dataset class for Vimeo-90k that returns GOP sequences
class Vimeo90kGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, septuplet_list, transform=None, crop_size=256, gop_size=7):
        """
        Args:
            root_dir (string): Directory with all the images.
            septuplet_list (string): Path to the file with list of septuplets.
            transform (callable, optional): Optional transform to be applied on a sample.
            crop_size (int): Size of the random crop.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.septuplet_list = []

        with open(septuplet_list, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.septuplet_list.append(line.strip())

    def __len__(self):
        return len(self.septuplet_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        septuplet_name = self.septuplet_list[idx]
        frames = []

        # Load frames
        for i in range(1, 8):  # Vimeo-90k septuplet has 7 frames
            img_name = os.path.join(self.root_dir, septuplet_name, f'im{i}.png')
            image = Image.open(img_name).convert('RGB')
            frames.append(image)

        # Apply random crop to the same location for all frames
        if self.crop_size:
            width, height = frames[0].size
            if width > self.crop_size and height > self.crop_size:
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in frames]

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        # For each frame, assign a positional index to help model understand GOP structure
        # First frame is position 0, then 1, 2, etc.
        # We'll use modulo GOP size to indicate the position in GOP
        frame_indices = list(range(len(frames)))

        # Random offset to simulate different positions in GOP
        offset = random.randint(0, self.gop_size - 1)
        frame_indices = [(i + offset) for i in frame_indices]

        return frames, frame_indices


# Define a dataset class for UVG that returns GOP sequences
class UVGGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, crop_size=None, gop_size=12):
        """
        Args:
            root_dir (string): Directory containing UVG video frames.
            transform (callable, optional): Optional transform to be applied on a sample.
            crop_size (int, optional): Optional crop size. If None, no cropping is performed.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.video_sequences = []

        # UVG videos - all 16 sequences
        video_names = [
            'Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide',
            'ChallengeRace', 'FoodMarket', 'Lips', 'RollerCoaster', 'SkateboardingTrick',
            'Squirrel', 'TallBuildings', 'ToddlerFountain', 'Tractor'
        ]

        # Get sequences of frames for each video
        for video_name in video_names:
            video_dir = os.path.join(root_dir, video_name)
            if os.path.isdir(video_dir):
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])

                # Divide frames into sequences of length gop_size (or maximum available)
                for i in range(0, len(frames), gop_size):
                    seq_frames = frames[i:min(i + gop_size, len(frames))]
                    if len(seq_frames) >= 2:  # Need at least 2 frames for P-frame training
                        self.video_sequences.append({
                            'video': video_name,
                            'frames': seq_frames
                        })

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.video_sequences[idx]
        video_name = sequence['video']
        frame_names = sequence['frames']

        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(self.root_dir, video_name, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        # Apply center crop if crop_size is specified
        if self.crop_size:
            width, height = frames[0].size
            if width > self.crop_size and height > self.crop_size:
                # Center crop
                left = (width - self.crop_size) // 2
                top = (height - self.crop_size) // 2
                frames = [img.crop((left, top, left + self.crop_size, top + self.crop_size)) for img in frames]

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        # Assign indices based on position in GOP
        frame_indices = list(range(len(frames)))

        return frames, frame_indices


def train_one_epoch(model, i_frame_model, train_loader, optimizer, device, lambda_value, stage, epoch, gop_size=12):
    """
    Train for one epoch with GOP structure awareness

    Args:
        gop_size: Group of Pictures size, typically 12 for non-HEVC videos, 10 for HEVC
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_bpp_y = 0
    total_bpp_z = 0
    total_bpp_mv_y = 0
    total_bpp_mv_z = 0
    n_batches = 0

    # We need frames organized in GOP structure, not random pairs
    progress_bar = tqdm(train_loader)

    # For training, we simulate GOP processing by treating each frame based on position
    for batch_idx, (frames, frame_indices) in enumerate(progress_bar):
        batch_loss = 0
        batch_mse = 0
        batch_bpp = 0

        # Process each sample in the batch
        for i in range(len(frames)):
            frame_sequence = frames[i]  # Get sequence of frames for this sample
            indices = frame_indices[i]  # Get corresponding indices

            # Convert to proper device
            frame_sequence = [f.to(device) for f in frame_sequence]

            # Determine if this is an I-frame or P-frame based on GOP position
            # First frame in GOP is I-frame, rest are P-frames
            ref_frame = None

            for j, (idx, frame) in enumerate(zip(indices, frame_sequence)):
                if idx % gop_size == 0:  # I-frame
                    # Process as I-frame (independently)
                    with torch.no_grad():  # Don't need gradients for I-frame during P-frame training
                        i_frame_result = i_frame_model(frame)
                        ref_frame = i_frame_result["x_hat"]
                else:  # P-frame
                    if ref_frame is None:
                        # If we don't have a reference frame yet, skip this frame
                        continue

                    # Process as P-frame using the previous frame as reference
                    # Control parameter freezing based on stage
                    if stage in [2, 3]:  # Freeze MV generation part in stages 2 and 3
                        for param in model.opticFlow.parameters():
                            param.requires_grad = False
                        for param in model.mvEncoder.parameters():
                            param.requires_grad = False
                        for param in model.mvDecoder_part1.parameters():
                            param.requires_grad = False
                        for param in model.mvDecoder_part2.parameters():
                            param.requires_grad = False
                    else:  # Unfreeze MV generation part in stages 1 and 4
                        for param in model.opticFlow.parameters():
                            param.requires_grad = True
                        for param in model.mvEncoder.parameters():
                            param.requires_grad = True
                        for param in model.mvDecoder_part1.parameters():
                            param.requires_grad = True
                        for param in model.mvDecoder_part2.parameters():
                            param.requires_grad = True

                    # Forward pass for P-frame
                    result = model(ref_frame, frame, training=True, stage=stage)

                    # Update reference frame for next frame
                    ref_frame = result["recon_image"].detach()  # Detach to prevent gradient propagation through time

                    # Accumulate loss
                    loss = result["loss"]
                    batch_loss += loss.item()

                    # Backward pass for each P-frame
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Gather metrics
                    batch_mse += nn.MSELoss()(result["recon_image"], frame).item()
                    batch_bpp += result["bpp"].item()
                    if "bpp_y" in result:
                        total_bpp_y += result["bpp_y"].item()
                    if "bpp_z" in result:
                        total_bpp_z += result["bpp_z"].item()
                    if "bpp_mv_y" in result:
                        total_bpp_mv_y += result["bpp_mv_y"].item()
                    if "bpp_mv_z" in result:
                        total_bpp_mv_z += result["bpp_mv_z"].item()

                    n_batches += 1

        # Update statistics
        if n_batches > 0:
            total_loss += batch_loss
            total_mse += batch_mse
            total_bpp += batch_bpp

            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch} Stage {stage} | "
                f"Loss: {total_loss / n_batches:.4f}, "
                f"MSE: {total_mse / n_batches:.6f}, "
                f"BPP: {total_bpp / n_batches:.4f}"
            )

    # Calculate epoch statistics
    if n_batches > 0:
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_psnr = -10 * math.log10(avg_mse)
        avg_bpp = total_bpp / n_batches
        avg_bpp_y = total_bpp_y / n_batches
        avg_bpp_z = total_bpp_z / n_batches
        avg_bpp_mv_y = total_bpp_mv_y / n_batches
        avg_bpp_mv_z = total_bpp_mv_z / n_batches
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0
        avg_bpp_y = 0
        avg_bpp_z = 0
        avg_bpp_mv_y = 0
        avg_bpp_mv_z = 0

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp,
        "bpp_y": avg_bpp_y,
        "bpp_z": avg_bpp_z,
        "bpp_mv_y": avg_bpp_mv_y,
        "bpp_mv_z": avg_bpp_mv_z
    }


def evaluate(model, i_frame_model, test_loader, device, stage, gop_size=12):
    """
    Evaluate model on test set with GOP structure awareness

    Args:
        gop_size: Group of Pictures size, typically 12 for non-HEVC videos, 10 for HEVC
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    n_frames = 0

    with torch.no_grad():
        for batch_idx, (frames, frame_indices) in enumerate(test_loader):
            # Process each sample in the batch
            for i in range(len(frames)):
                frame_sequence = frames[i]  # Get sequence of frames for this sample
                indices = frame_indices[i]  # Get corresponding indices

                # Convert to proper device
                frame_sequence = [f.to(device) for f in frame_sequence]

                # Initialize reference frame
                ref_frame = None

                # Process each frame in the sequence according to GOP structure
                for j, (idx, frame) in enumerate(zip(indices, frame_sequence)):
                    if idx % gop_size == 0:  # I-frame
                        # Process as I-frame
                        i_frame_result = i_frame_model(frame.unsqueeze(0))
                        ref_frame = i_frame_result["x_hat"]

                        # Skip metrics for I-frames (we're mainly interested in P-frame performance)
                    else:  # P-frame
                        if ref_frame is None:
                            # If we don't have a reference frame yet, skip this frame
                            continue

                        # Process as P-frame
                        result = model(ref_frame, frame.unsqueeze(0), training=False, stage=stage)

                        # Update reference frame for next iteration
                        ref_frame = result["recon_image"]

                        # Calculate metrics for P-frames
                        loss = result["loss"]
                        total_loss += loss.item()

                        mse = nn.MSELoss()(result["recon_image"], frame.unsqueeze(0))
                        total_mse += mse.item()

                        total_bpp += result["bpp"].item()

                        n_frames += 1

    # Calculate average statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse)
        avg_bpp = total_bpp / n_frames
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp
    }


def main():
    parser = argparse.ArgumentParser(description='DCVC Training')
    parser.add_argument('--vimeo_dir', type=str, required=True, help='Path to Vimeo-90k dataset')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', help='I-frame model name')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (0-3) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4], help='Training stage (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='psnr', choices=['psnr', 'ms-ssim'],
                        help='Model type: psnr or ms-ssim')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--previous_stage_checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous stage to resume from')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    # Add GOP size argument
    parser.add_argument('--gop_size', type=int, default=12, help='GOP size (12 for non-HEVC, 10 for HEVC)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA devices
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create checkpoint and log directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Remove any old dataset class definitions still in the code
    # Vimeo90kDataset and UVGDataset should be replaced completely

    # Create training dataset with GOP structure
    train_dataset = Vimeo90kGOPDataset(
        root_dir=args.vimeo_dir,
        septuplet_list=args.septuplet_list,
        transform=transform,
        crop_size=args.crop_size,
        gop_size=args.gop_size
    )

    # Create a custom collate function to handle variable-length sequences
    def custom_collate_fn(batch):
        frames_list = [item[0] for item in batch]
        indices_list = [item[1] for item in batch]
        return frames_list, indices_list

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Create test dataset (UVG) with GOP structure
    test_dataset = UVGGOPDataset(
        root_dir=args.uvg_dir,
        transform=transform,
        crop_size=args.crop_size if args.crop_size <= 1024 else 1024,  # UVG is 1080p, so crop to 1024 if needed
        gop_size=args.gop_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Make script print info about UVG dataset at start
    print(f"UVG dataset loaded with {len(test_dataset)} frame pairs.")

    # Load I-frame model
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_load_checkpoint).eval()
    i_frame_model = i_frame_model.to(device)

    print(
        f"Training model with lambda = {args.lambda_value}, quality_index = {args.quality_index}, stage = {args.stage}")

    # Initialize DCVC model
    model = DCVC_net(lmbda=args.lambda_value)
    model = model.to(device)

    # Initialize optimizer
    if args.stage == 4 and args.previous_stage_checkpoint:
        # Use lower learning rate for stage 4
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load from previous stage checkpoint if specified
    start_epoch = 0
    if args.previous_stage_checkpoint:
        print(f"Loading model from previous stage checkpoint: {args.previous_stage_checkpoint}")
        checkpoint = torch.load(args.previous_stage_checkpoint, map_location=device)
        model.load_dict(checkpoint)  # Use load_dict method as defined in DCVC_net

    # Log file
    stage_descriptions = {
        1: "Warm up MV generation part",
        2: "Train other modules",
        3: "Train with bit cost",
        4: "End-to-end training"
    }

    log_file = os.path.join(args.log_dir,
                            f'train_log_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_{args.model_type}.txt')
    with open(log_file, 'a') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lambda value: {args.lambda_value}\n")
        f.write(f"Quality index: {args.quality_index}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Stage: {args.stage} ({stage_descriptions[args.stage]})\n")
        f.write(f"I-frame model: {args.i_frame_model_path}\n")
        if args.previous_stage_checkpoint:
            f.write(f"Previous stage checkpoint: {args.previous_stage_checkpoint}\n")
        f.write(f"UVG dataset: {len(test_dataset)} frame pairs\n")
        f.write("=" * 80 + "\n")

    # Track best model
    best_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch with GOP awareness
        train_stats = train_one_epoch(model, i_frame_model, train_loader, optimizer, device, args.lambda_value,
                                      args.stage, epoch + 1, args.gop_size)

        # Evaluate on test set with GOP awareness
        test_stats = evaluate(model, i_frame_model, test_loader, device, args.stage, args.gop_size)

        # Log results
        with open(log_file, 'a') as f:
            f.write(f"Stage {args.stage}, Epoch {epoch + 1}/{args.epochs}:\n")
            f.write(f"  Train Loss: {train_stats['loss']:.6f}\n")
            f.write(f"  Train MSE: {train_stats['mse']:.6f}\n")
            f.write(f"  Train PSNR: {train_stats['psnr']:.4f}\n")
            f.write(f"  Train BPP: {train_stats['bpp']:.6f}\n")
            if 'bpp_y' in train_stats:
                f.write(f"  Train BPP_y: {train_stats['bpp_y']:.6f}\n")
            if 'bpp_z' in train_stats:
                f.write(f"  Train BPP_z: {train_stats['bpp_z']:.6f}\n")
            if 'bpp_mv_y' in train_stats:
                f.write(f"  Train BPP_mv_y: {train_stats['bpp_mv_y']:.6f}\n")
            if 'bpp_mv_z' in train_stats:
                f.write(f"  Train BPP_mv_z: {train_stats['bpp_mv_z']:.6f}\n")
            f.write(f"  Test Loss: {test_stats['loss']:.6f}\n")
            f.write(f"  Test MSE: {test_stats['mse']:.6f}\n")
            f.write(f"  Test PSNR: {test_stats['psnr']:.4f}\n")
            f.write(f"  Test BPP: {test_stats['bpp']:.6f}\n")

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest.pth'
        )
        torch.save(model.state_dict(), latest_checkpoint_path)

        # Save best checkpoint if current test loss is the best so far
        if test_stats['loss'] < best_loss:
            best_loss = test_stats['loss']
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best.pth'
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved with test loss: {best_loss:.6f}")

        print(f"Epoch {epoch + 1}/{args.epochs} completed. Latest checkpoint saved.")

    # Save final model for this stage
    final_model_path = os.path.join(
        args.checkpoint_dir,
        f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}.pth'
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model for stage {args.stage} saved to {final_model_path}")

    # If this is the final stage (4), also save with the standard naming convention
    if args.stage == 4:
        standard_model_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_quality_{args.quality_index}_{args.model_type}.pth'
        )
        torch.save(model.state_dict(), standard_model_path)
        print(f"Final model (standard name) saved to {standard_model_path}")

    with open(log_file, 'a') as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    print(f"Training completed for stage {args.stage}!")


if __name__ == '__main__':
    main()

# python train_dcvc.py \
#   --vimeo_dir /path/to/vimeo_90k/train \
#   --septuplet_list /path/to/vimeo_90k/sep_trainlist.txt \
#   --uvg_dir /path/to/uvg_dataset \
#   --i_frame_model_path checkpoints/cheng2020-anchor-3-e49be189.pth.tar \
#   --lambda_value 256 \
#   --quality_index 0 \
#   --stage 1 \
#   --epochs 5 \
#   --gop_size 12 \  # Use 10 for HEVC test videos
#   --model_type psnr \
#   --batch_size 4