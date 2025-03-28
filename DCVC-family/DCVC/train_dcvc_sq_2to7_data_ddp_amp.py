import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler  # Import AMP components
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
import torchvision.transforms as transforms
import time
import builtins
from timm.utils import unwrap_model

# Import the model
from src.models.DCVC_net import DCVC_net
from src.zoo.image import model_architectures as architectures


# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define a dataset class for Vimeo-90k that returns GOP sequences
class Vimeo90kGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, septuplet_list, transform=None, crop_size=256, gop_size=7, shuffle_frames=True):
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
        self.shuffle_frames = shuffle_frames
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


        # Random shuffle frame order if enabled
        if self.shuffle_frames:
            # Create a list of indices and shuffle it
            frame_indices = list(range(len(frames)))
            random.shuffle(frame_indices)
            
            # Reorder frames according to shuffled indices
            frames = [frames[i] for i in frame_indices]
            
            # Option: You could also return the shuffle indices if needed for reconstruction
            # return torch.stack(frames), frame_indices

        return torch.stack(frames)  # Return frames as a single tensor [S, C, H, W]


# Define a dataset class for UVG that returns GOP sequences
class UVGGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, gop_size=12):
        """
        Args:
            root_dir (string): Directory containing UVG video frames.
            transform (callable, optional): Optional transform to be applied on a sample.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.gop_size = gop_size
        self.video_sequences = []

        # UVG videos
        video_names = [
            'Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV', 
            'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV', 
            'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV', 
            'YachtRide_1920x1024_120fps_420_8bit_YUV'
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

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        return torch.stack(frames)  # Return frames as a single tensor [S, C, H, W]


def train_one_epoch_fully_batched(model, i_frame_model, train_loader, optimizer, device, stage, epoch, 
                                 gradient_accumulation_steps=1, finetune=False, distributed=False, world_size=1,
                                 scaler=None, use_amp=False):  # Added AMP parameters
    """
    Train for one epoch with fully batched processing for GOP sequences.
    
    Args:
        gop_size: Group of Pictures size (7 for Vimeo90k)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        scaler: GradScaler for Automatic Mixed Precision
        use_amp: Flag to enable autocast for mixed precision
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    # Control parameter freezing based on stage - only do this once per epoch
    if stage in [2, 3]:  # Freeze MV generation part in stages 2 and 3
        for param in unwrap_model(model).opticFlow.parameters():
            param.requires_grad = False
        for param in unwrap_model(model).mvEncoder.parameters():
            param.requires_grad = False
        for param in unwrap_model(model).mvDecoder_part1.parameters():
            param.requires_grad = False
        for param in unwrap_model(model).mvDecoder_part2.parameters():
            param.requires_grad = False
    else:  # Unfreeze MV generation part in stages 1 and 4
        for param in unwrap_model(model).opticFlow.parameters():
            param.requires_grad = True
        for param in unwrap_model(model).mvEncoder.parameters():
            param.requires_grad = True
        for param in unwrap_model(model).mvDecoder_part1.parameters():
            param.requires_grad = True
        for param in unwrap_model(model).mvDecoder_part2.parameters():
            param.requires_grad = True
    
    # Process batches of GOP sequences
    # In distributed setting, we need to decide whether to use tqdm on all processes
    if not distributed or (distributed and dist.get_rank() == 0):
        train_loader = tqdm(train_loader)
    
    for batch_idx, batch_frames in enumerate(train_loader):
        batch_size = batch_frames.size(0)
        seq_length = batch_frames.size(1)  # Number of frames in sequence (S dimension)
        batch_loss = 0
        
        # Initialize reference frames for the batch
        reference_frames = None
        
        if finetune and stage == 4:
            # Process each frame position in the sequence
            for frame_pos in range(seq_length):
                # Get all frames at this position from all sequences in the batch
                current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                
                if frame_pos == 0:  # First frame (I-frame) in each sequence
                    # Process all I-frames in the batch together
                    with torch.no_grad():  # Don't train I-frame model
                        # Use autocast for I-frame if using AMP
                        with autocast(enabled=use_amp):
                            i_frame_results = i_frame_model(current_frames)
                        reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                else:  # P-frames
                    # Zero gradients for each frame position
                    optimizer.zero_grad()

                    # Process all P-frames in the batch with their corresponding reference frames
                    # Use autocast for forward pass if using AMP
                    with autocast(enabled=use_amp):
                        result = model(reference_frames, current_frames, training=True, stage=stage)
                        loss = result["loss"]
                    
                    # Update reference frames for next position with the reconstructed frames
                    reference_frames = result["recon_image"].detach()  # Shape: [B, C, H, W]
                    
                    # Handle backward with scaler if using AMP
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # Collect statistics
                    batch_loss += result["loss"].item()
                    total_mse += result["mse_loss"].item() * batch_size  # Account for all frames in batch
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    
                    n_frames += batch_size  # Count all frames in batch
        else:
            # Two frame training
            # Process each frame position in the sequence
            for frame_pos in range(seq_length):
                if frame_pos == 0:  # First frame (I-frame) in each sequence
                    # skip I-frame processing
                    continue
                else:  # P-frames
                    previous_frames = batch_frames[:, frame_pos - 1, :, :, :].to(device)  # Shape: [B, C, H, W]
                    current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                    # process previous frames by I-frame model
                    with torch.no_grad():  # Don't train I-frame model
                        # Use autocast for I-frame if using AMP
                        with autocast(enabled=use_amp):
                            i_frame_results = i_frame_model(previous_frames)
                        reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                    
                    # Zero gradients for each frame position
                    optimizer.zero_grad()

                    # Process all P-frames in the batch with their corresponding reference frames
                    # Use autocast for forward pass if using AMP
                    with autocast(enabled=use_amp):
                        result = model(reference_frames, current_frames, training=True, stage=stage)
                        loss = result["loss"]

                    # Handle backward with scaler if using AMP
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # Collect statistics
                    batch_loss += result["loss"].item()
                    total_mse += result["mse_loss"].item() * batch_size  # Account for all frames in batch
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size

                    n_frames += batch_size  # Count all frames in batch
        
        # Update total loss
        total_loss += batch_loss
        
        # Update progress bar (only for rank 0 in distributed training)
        if not distributed or (distributed and dist.get_rank() == 0):
            if n_frames > 0:
                train_loader.set_description(
                    f"Epoch {epoch} Stage {stage} | "
                    f"Loss: {total_loss / n_frames:.4f}, "
                    f"MSE: {total_mse / n_frames:.6f}, "
                    f"BPP: {total_bpp / n_frames:.4f}, "
                    f"PSNR: {total_psnr / n_frames:.4f}"
                )
    
    # Aggregate statistics across all processes in distributed mode
    if distributed:
        # Create tensors to hold the values
        stats_tensor = torch.tensor([total_loss, total_mse, total_bpp, total_psnr, n_frames], dtype=torch.float64, device=device)
        
        # All-reduce to sum the tensors across all processes
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        # Extract values back
        total_loss, total_mse, total_bpp, total_psnr, n_frames = stats_tensor.tolist()
    
    # Calculate epoch statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else 100
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
        "bpp": avg_bpp,
    }


def evaluate_fully_batched(model, i_frame_model, test_loader, device, stage, finetune=False, distributed=False, 
                           use_amp=False):  # AMP parameter defaults to False for evaluation
    """
    Evaluate model using fully batched processing for GOP sequences
    
    Args:
        gop_size: Group of Pictures size (12 for UVG)
        use_amp: Flag to enable autocast for mixed precision
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    with torch.no_grad():
        for batch_frames in test_loader:
            batch_size = batch_frames.size(0)
            seq_length = batch_frames.size(1)  # Number of frames in sequence
            
            # Initialize reference frames
            reference_frames = None
            
            if finetune and stage == 4: 
                # Real coding order
                # Process each frame position in the sequence
                for frame_pos in range(seq_length):
                    # Get all frames at this position from all sequences in the batch
                    current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                    
                    if frame_pos == 0:  # I-frames
                        # Process all I-frames in the batch together
                        # Use autocast for I-frame if using AMP
                        with autocast(enabled=use_amp):
                            i_frame_results = i_frame_model(current_frames)
                        reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                    else:  # P-frames
                        # Process all P-frames in the batch with their corresponding reference frames
                        # Use autocast for P-frames if using AMP
                        with autocast(enabled=use_amp):
                            result = model(reference_frames, current_frames, training=False, stage=stage)
                        
                        # Update reference frames for next position
                        reference_frames = result["recon_image"]  # Shape: [B, C, H, W]
                        
                        # Collect statistics
                        total_loss += result["loss"].item() * batch_size
                        total_mse += result["mse_loss"].item() * batch_size
                        total_bpp += result["bpp_train"].item() * batch_size
                        total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                        
                        n_frames += batch_size
            else:
                # Two frame evaluation
                # Process each frame position in the sequence
                for frame_pos in range(seq_length):
                    if frame_pos == 0:
                        continue
                    else:
                        previous_frames = batch_frames[:, frame_pos - 1, :, :, :].to(device)
                        current_frames = batch_frames[:, frame_pos, :, :, :].to(device)
                        # process previous frames by I-frame model
                        # Use autocast for I-frame if using AMP
                        with autocast(enabled=use_amp):
                            i_frame_results = i_frame_model(previous_frames)
                        reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                        
                        # Process all P-frames in the batch with their corresponding reference frames
                        # Use autocast for P-frames if using AMP
                        with autocast(enabled=use_amp):
                            result = model(reference_frames, current_frames, training=False, stage=stage)
                        
                        # Collect statistics
                        total_loss += result["loss"].item() * batch_size
                        total_mse += result["mse_loss"].item() * batch_size
                        total_bpp += result["bpp"].item() * batch_size
                        total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                        n_frames += batch_size
    
    # Aggregate statistics across all processes in distributed mode
    if distributed:
        # Create tensors to hold the values
        stats_tensor = torch.tensor([total_loss, total_mse, total_bpp, total_psnr, n_frames], dtype=torch.float64, device=device)
        
        # All-reduce to sum the tensors across all processes
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        # Extract values back
        total_loss, total_mse, total_bpp, total_psnr, n_frames = stats_tensor.tolist()
    
    # Calculate average statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else 100
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
    parser = argparse.ArgumentParser(description='DCVC Training with Distributed Data Parallel and AMP')
    parser.add_argument('--vimeo_dir', type=str, required=True, help='Path to Vimeo-90k dataset')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', help='I-frame model name')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_data_ddp_amp', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs_data_ddp_amp', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (0-3) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4], help='Training stage (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='4,5,6,7', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading per GPU')
    parser.add_argument('--model_type', type=str, default='psnr', choices=['psnr', 'ms-ssim'],
                        help='Model type: psnr or ms-ssim')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--previous_stage_checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous stage to resume from')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    
    # Add arguments for resume training and SpyNet initialization
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--spynet_checkpoint', type=str, default=None, 
                       help='Path to SpyNet pretrained weights to initialize motion estimation network')
    parser.add_argument('--spynet_from_dcvc_checkpoint', type=str, default=None,
                        help='Path to DCVC checkpoint to initialize SpyNet weights')
    
    # Add learning rate scheduler arguments
    parser.add_argument('--lr_scheduler', type=str, default='step', 
                        choices=['step', 'multistep', 'cosine', 'plateau', 'none'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=5, 
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, 
                        help='Gamma for StepLR and MultiStepLR schedulers (multiplicative factor)')
    parser.add_argument('--lr_milestones', type=str, default='5,10,15', 
                        help='Comma-separated milestone epochs for MultiStepLR scheduler')
    parser.add_argument('--lr_min_factor', type=float, default=0.01, 
                        help='Minimum lr factor for CosineAnnealingLR scheduler (as a fraction of initial lr)')
    parser.add_argument('--lr_patience', type=int, default=2, 
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs with linearly increasing LR')
    
    # Add gradient accumulation option
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients (for larger effective batch sizes)')
    # torch.compile
    parser.add_argument('--compile', action='store_true', help='Compile the model')

    # finetune indicator
    parser.add_argument('--finetune', action='store_true', help='Finetune the model')
    
    # DDP specific arguments
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank for distributed training')
    parser.add_argument('--sync_bn', action='store_true', 
                        help='Use Synchronized BatchNorm in distributed training')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='Find unused parameters in DDP (needed for some model architectures)')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='Distributed backend')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training')
    
    # Add AMP arguments
    parser.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision for training')
    parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                       help='''Data type for AMP: 
                             - float16: Better speed, less precision (10-bit mantissa), limited dynamic range
                             - bfloat16: Better numerical stability (8-bit exponent like FP32), less precision (7-bit mantissa)
                             bfloat16 is recommended for more complex models but requires hardware support (e.g., A100, H100 GPUs)''')
                       
    # Note: The following is added for backward compatibility with earlier AMP implementations
    # but PyTorch's native AMP doesn't use these levels explicitly
    parser.add_argument('--opt_level', type=str, default='O1', 
                       help='''Legacy optimization level concept from NVIDIA Apex:
                             - O0: FP32 training (no mixed precision)
                             - O1: Mixed precision (FP16 where beneficial, FP32 elsewhere) - PyTorch native AMP behavior
                             - O2: "Almost FP16" - uses FP16 wherever possible with FP32 master weights
                             - O3: Pure FP16 (can be numerically unstable)
                             With native PyTorch AMP, we're effectively always using O1 behavior''')

    args = parser.parse_args()

    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        # For local testing (when not using Slurm or torchrun)
        args.rank = args.local_rank
        args.world_size = torch.cuda.device_count()

    # Set up distributed process group
    if args.local_rank != -1:
        if args.local_rank == 0:
            print(f"Starting distributed training with {args.world_size} GPUs")
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()  # Synchronize processes before starting
        
        # Suppress print statements for non-master processes
        if args.rank != 0:
            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA devices
    if args.cuda:
        device = torch.device(f'cuda:{args.local_rank}' if args.local_rank != -1 else 'cuda')
    else:
        device = torch.device('cpu')

    # Create checkpoint and log directories (only on rank 0)
    if args.rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create training dataset with GOP structure
    train_dataset = Vimeo90kGOPDataset(
        root_dir=args.vimeo_dir,
        septuplet_list=args.septuplet_list,
        transform=transform,
        crop_size=args.crop_size,
        gop_size=7  # Vimeo90k has 7 frames per sequence
    )
    
    # Create distributed sampler for the training dataset
    if args.local_rank != -1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
        shuffle = False  # Sampler already handles shuffling
    else:
        train_sampler = None
        shuffle = True
    
    # Create dataloader with distributed sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True  # Important for DDP to avoid uneven batch sizes
    )

    # Create test dataset (UVG) with GOP structure
    test_dataset = UVGGOPDataset(
        root_dir=args.uvg_dir,
        transform=transform,
        gop_size=12
    )
    
    # For evaluation, we can either use a regular sampler or a distributed one
    # Here we use a regular one since the test set is small and each GPU should evaluate on all data
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Make script print info about UVG dataset at start (only on rank 0)
    if args.rank == 0:
        print(f"UVG dataset loaded with {len(test_dataset)} sequences.")

    # Load I-frame model
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_load_checkpoint).eval()
    i_frame_model = i_frame_model.to(device)
    
    # Compile I-frame model if needed
    if args.compile:
        if args.rank == 0:
            print("Compiling the I-frame model...")
        i_frame_model = torch.compile(i_frame_model)

    if args.rank == 0:
        print(
            f"Training model with lambda = {args.lambda_value}, quality_index = {args.quality_index}, stage = {args.stage}")

    # Initialize DCVC model
    model = DCVC_net(lmbda=args.lambda_value)
    model = model.to(device)
    
    # Synchronize batch normalization across all processes if specified
    if args.sync_bn and args.local_rank != -1:
        print("Using Synchronized BatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Check for SpyNet initialization (do this before DDP wrapping)
    if args.spynet_checkpoint:
        if args.rank == 0:
            print(f"Initializing motion estimation network with pretrained SpyNet weights: {args.spynet_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_checkpoint, map_location=device)
        
        spynet_state_dict = spynet_checkpoint['state_dict']
        model.opticFlow.load_state_dict(spynet_state_dict, strict=True)
        if args.rank == 0:
            print("Loaded SpyNet weights directly into opticFlow component")
    
    if args.spynet_from_dcvc_checkpoint:
        if args.rank == 0:
            print(f"Initializing motion estimation network with SpyNet weights from DCVC checkpoint: {args.spynet_from_dcvc_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_from_dcvc_checkpoint, map_location=device)

        # Extract only SpyNet weights
        spynet_state_dict = {}
        for key, value in spynet_checkpoint.items():
            if key.startswith('opticFlow.'):
                spynet_state_dict[key.replace('opticFlow.', '')] = value
        
        model.opticFlow.load_state_dict(spynet_state_dict, strict=True)
        if args.rank == 0:
            print("Loaded SpyNet weights from DCVC checkpoint into opticFlow component")
    
    # Resume training from checkpoint (before DDP wrapping)
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if args.rank == 0:
            print(f"Resuming training from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if this is a state_dict only or a complete checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_loss = checkpoint['best_loss']
                if args.rank == 0:
                    print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.6f}")
            else:
                # State dict only
                model.load_state_dict(checkpoint)
                if args.rank == 0:
                    print("Loaded model weights only (no training state)")
        except Exception as e:
            if args.rank == 0:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch")
    # Load from previous stage checkpoint if specified
    elif args.previous_stage_checkpoint:
        if args.rank == 0:
            print(f"Loading model from previous stage checkpoint: {args.previous_stage_checkpoint}")
        try:
            checkpoint = torch.load(args.previous_stage_checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
            if args.rank == 0:
                print("Successfully loaded model from previous stage")
        except Exception as e:
            if args.rank == 0:
                print(f"Error loading previous stage checkpoint: {e}")
                print("Starting training from scratch")
    
    # Wrap the model with DDP
    if args.local_rank != -1:
        model = DDP(model, 
                   device_ids=[args.local_rank], 
                   output_device=args.local_rank,
                   find_unused_parameters=args.find_unused_parameters,
                   broadcast_buffers=False)  # Sometimes False works better

    # Compile the model if specified (after DDP wrapping)
    if args.compile:
        if args.rank == 0:
            print("Compiling the model...")
        model = torch.compile(model)

    # Initialize optimizer
    if args.stage == 4 and args.previous_stage_checkpoint:
        # Use lower learning rate for stage 4
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize AMP gradient scaler
    scaler = GradScaler() if args.amp else None
    
    # If resuming training with a complete checkpoint, also load scaler state
    if args.resume and isinstance(checkpoint, dict) and 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if args.rank == 0:
            print("Resumed scaler state for AMP")

    # Initialize learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'multistep':
        milestones = [int(m) for m in args.lr_milestones.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * args.lr_min_factor
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience, verbose=(args.rank == 0)
        )
    else:  # 'none' or any other value
        scheduler = None

    # Create a warmup scheduler if warmup epochs are specified
    if args.lr_warmup_epochs > 0 and scheduler is not None and args.lr_scheduler != 'plateau':
        from torch.optim.lr_scheduler import LambdaLR
        
        # Create warmup scheduler
        def warmup_lambda(epoch):
            if epoch < args.lr_warmup_epochs:
                return epoch / args.lr_warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        warmup_scheduler = None

    # If resuming training with a complete checkpoint, also load scheduler state
    if args.resume and isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.rank == 0:
            print("Resumed scheduler state")

    #resume optimizer state if resuming training
    if args.resume and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.rank == 0:
            print("Resumed optimizer state")

    # resume scaler state if resuming training
    if args.resume and isinstance(checkpoint, dict) and 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if args.rank == 0:
            print("Resumed scaler state")

    # Log file (only on rank 0)
    if args.rank == 0:
        stage_descriptions = {
            1: "Warm up MV generation part",
            2: "Train other modules",
            3: "Train with bit cost",
            4: "End-to-end training"
        }

        log_file = os.path.join(args.log_dir,
                                f'train_log_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_{args.model_type}_amp.txt')
        with open(log_file, 'a') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Distributed training with {args.world_size} GPUs\n")
            f.write(f"Lambda value: {args.lambda_value}\n")
            f.write(f"Quality index: {args.quality_index}\n")
            f.write(f"Model type: {args.model_type}\n")
            f.write(f"Stage: {args.stage} ({stage_descriptions[args.stage]})\n")
            f.write(f"I-frame model: {args.i_frame_model_path}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"LR Scheduler: {args.lr_scheduler}\n")
            f.write(f"Batch size per GPU: {args.batch_size}\n")
            f.write(f"Total batch size: {args.batch_size * args.world_size}\n")
            f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
            f.write(f"Effective batch size per GPU: {args.batch_size * args.gradient_accumulation_steps}\n")
            f.write(f"Total effective batch size: {args.batch_size * args.world_size * args.gradient_accumulation_steps}\n")
            f.write(f"AMP enabled: {args.amp}\n")
            if args.amp:
                f.write(f"  AMP opt level: {args.opt_level}\n")
                f.write(f"  AMP dtype: {args.amp_dtype}\n")
            if args.sync_bn:
                f.write("Using Synchronized BatchNorm\n")
            if args.lr_scheduler != 'none':
                if args.lr_scheduler == 'step':
                    f.write(f"  Step size: {args.lr_step_size}, Gamma: {args.lr_gamma}\n")
                elif args.lr_scheduler == 'multistep':
                    f.write(f"  Milestones: {args.lr_milestones}, Gamma: {args.lr_gamma}\n")
                elif args.lr_scheduler == 'cosine':
                    f.write(f"  Min LR factor: {args.lr_min_factor}\n")
                elif args.lr_scheduler == 'plateau':
                    f.write(f"  Patience: {args.lr_patience}, Factor: {args.lr_gamma}\n")
                if args.lr_warmup_epochs > 0:
                    f.write(f"  Warmup epochs: {args.lr_warmup_epochs}\n")
            if args.previous_stage_checkpoint:
                f.write(f"Previous stage checkpoint: {args.previous_stage_checkpoint}\n")
            if args.resume:
                f.write(f"Resuming from checkpoint: {args.resume}\n")
                f.write(f"Starting from epoch: {start_epoch}\n")
            if args.spynet_checkpoint:
                f.write(f"SpyNet initialization: {args.spynet_checkpoint}\n")
            if args.spynet_from_dcvc_checkpoint:
                f.write(f"SpyNet from DCVC initialization: {args.spynet_from_dcvc_checkpoint}\n")
            f.write(f"UVG dataset: {len(test_dataset)} sequences\n")
            f.write(f"Using fully batched GOP processing with parallel P-frame batch processing\n")
            f.write("=" * 80 + "\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set distributed sampler epoch
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        
        # Log current learning rate (only on rank 0)
        current_lr = optimizer.param_groups[0]['lr']
        if args.rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}\n")

        # Apply warmup scheduler if in warmup phase
        if warmup_scheduler is not None and epoch < args.lr_warmup_epochs:
            warmup_scheduler.step()
        
        # Train one epoch with fully batched GOP processing and AMP
        train_stats = train_one_epoch_fully_batched(
            model, i_frame_model, train_loader, optimizer, device,
            args.stage, epoch + 1,
            args.gradient_accumulation_steps, args.finetune,
            distributed=(args.local_rank != -1), world_size=args.world_size,
            scaler=scaler, use_amp=args.amp  # Pass scaler and AMP flag
        )

        # Evaluate on test set with fully batched GOP processing (using FP32 for accuracy)
        test_stats = evaluate_fully_batched(
            model, i_frame_model, test_loader, device, args.stage, args.finetune,
            distributed=(args.local_rank != -1),
            use_amp=False  # Use full precision for evaluation for more accurate metrics
        )

        # Step scheduler after training (different for ReduceLROnPlateau)
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(test_stats['loss'])
            elif epoch >= args.lr_warmup_epochs:  # Only step main scheduler after warmup
                scheduler.step()

        # Log results (only on rank 0)
        if args.rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"Stage {args.stage}, Epoch {epoch + 1}/{args.epochs}:\n")
                f.write(f"  Train Loss: {train_stats['loss']:.6f}\n")
                f.write(f"  Train MSE: {train_stats['mse']:.6f}\n")
                f.write(f"  Train PSNR: {train_stats['psnr']:.4f}\n")
                f.write(f"  Train BPP: {train_stats['bpp']:.6f}\n")
                if 'bpp_y' in train_stats and train_stats['bpp_y'] > 0:
                    f.write(f"  Train BPP_y: {train_stats['bpp_y']:.6f}\n")
                if 'bpp_z' in train_stats and train_stats['bpp_z'] > 0:
                    f.write(f"  Train BPP_z: {train_stats['bpp_z']:.6f}\n")
                if 'bpp_mv_y' in train_stats and train_stats['bpp_mv_y'] > 0:
                    f.write(f"  Train BPP_mv_y: {train_stats['bpp_mv_y']:.6f}\n")
                if 'bpp_mv_z' in train_stats and train_stats['bpp_mv_z'] > 0:
                    f.write(f"  Train BPP_mv_z: {train_stats['bpp_mv_z']:.6f}\n")
                f.write(f"  Test Loss: {test_stats['loss']:.6f}\n")
                f.write(f"  Test MSE: {test_stats['mse']:.6f}\n")
                f.write(f"  Test PSNR: {test_stats['psnr']:.4f}\n")
                f.write(f"  Test BPP: {test_stats['bpp']:.6f}\n")

            # Save latest checkpoint with training state for resuming (only on rank 0)
            latest_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest_amp.pth'
            )
            
            # Get model state dict accounting for DDP wrapper
            if args.local_rank != -1:
                model_state_dict = unwrap_model(model).state_dict()
            else:
                model_state_dict = model.state_dict()
            
            # Create save dictionary with all training state
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_stats['loss'],
                'best_loss': best_loss,
                'stage': args.stage,
                'quality_index': args.quality_index,
                'lambda_value': args.lambda_value
            }
            
            # Add scheduler state if present
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            
            # Add scaler state if using AMP
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            # Save the checkpoint
            torch.save(save_dict, latest_checkpoint_path)

            # Save best checkpoint if current test loss is the best so far
            if test_stats['loss'] < best_loss:
                best_loss = test_stats['loss']
                best_checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best_amp.pth'
                )
                # Save full training state for the best model too
                torch.save(save_dict, best_checkpoint_path)
                print(f"New best model saved with test loss: {best_loss:.6f}")

            print(f"Epoch {epoch + 1}/{args.epochs} completed. Latest checkpoint saved.")

    # Save final model for this stage (state_dict only for compatibility with original code)
    if args.rank == 0:
        final_model_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_amp.pth'
        )
        
        # Get model state dict accounting for DDP wrapper
        if args.local_rank != -1:
            model_state_dict = unwrap_model(model).state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save(model_state_dict, final_model_path)
        print(f"Final model for stage {args.stage} saved to {final_model_path}")

        # If this is the final stage (4), also save with the standard naming convention
        if args.stage == 4:
            standard_model_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_{args.model_type}_amp.pth'
            )
            torch.save(model_state_dict, standard_model_path)
            print(f"Final model (standard name) saved to {standard_model_path}")

        with open(log_file, 'a') as f:
            f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        print(f"Training completed for stage {args.stage}!")

    # Clean up distributed process group
    if args.local_rank != -1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()