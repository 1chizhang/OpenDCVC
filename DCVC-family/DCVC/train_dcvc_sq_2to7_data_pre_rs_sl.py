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
from timm.utils import unwrap_model

# Import the model
from src.models.DCVC_net import DCVC_net


# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Modified dataset class that returns pairs of (reference, current) frames directly
class Vimeo90kPairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, precomputed_dir, septuplet_list, transform=None, crop_size=256):
        """
        Args:
            root_dir (string): Directory with all the images.
            precomputed_dir (string): Directory with precomputed reference frames.
            septuplet_list (string): Path to the file with list of septuplets.
            transform (callable, optional): Optional transform to be applied on a sample.
            crop_size (int): Size of the random crop.
            shuffle_frames (bool): Whether to shuffle the frame pairs.
        """
        self.root_dir = root_dir
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        self.crop_size = crop_size
        self.septuplet_list = []
        self.frame_pairs = []

        with open(septuplet_list, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.septuplet_list.append(line.strip())
        
        # Generate pairs of (reference, current) frames
        for septuplet_name in self.septuplet_list:
            for i in range(0, 6):  # We have 6 possible pairs in a septuplet
                # Each pair consists of a reference frame and a target frame
                self.frame_pairs.append({
                    'septuplet': septuplet_name,
                    'ref_idx': i,      # Index of reference frame (0-5)
                    'curr_idx': i + 1  # Index of current frame (1-6)
                })
        

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.frame_pairs[idx]
        septuplet_name = pair['septuplet']
        ref_idx = pair['ref_idx']
        curr_idx = pair['curr_idx']
        
        # Load reference frame from precomputed directory
        ref_path = os.path.join(self.precomputed_dir, septuplet_name, f'ref{ref_idx+1}.png')
        ref_frame = Image.open(ref_path).convert('RGB')
        
        # Load current frame to be compressed
        curr_path = os.path.join(self.root_dir, septuplet_name, f'im{curr_idx+1}.png')
        curr_frame = Image.open(curr_path).convert('RGB')

        # Apply random crop to the same location for both frames
        if self.crop_size:
            width, height = curr_frame.size
            if width > self.crop_size and height > self.crop_size:
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                ref_frame = ref_frame.crop((x, y, x + self.crop_size, y + self.crop_size))
                curr_frame = curr_frame.crop((x, y, x + self.crop_size, y + self.crop_size))

        # Apply random flips for data augmentation
        if random.random() < 0.5:  # 50% chance to flip horizontally
            ref_frame = ref_frame.transpose(Image.FLIP_LEFT_RIGHT)
            curr_frame = curr_frame.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:  # 50% chance to flip vertically
            ref_frame = ref_frame.transpose(Image.FLIP_TOP_BOTTOM)
            curr_frame = curr_frame.transpose(Image.FLIP_TOP_BOTTOM)

        # Apply transform if provided
        if self.transform:
            ref_frame = self.transform(ref_frame)
            curr_frame = self.transform(curr_frame)

        return ref_frame, curr_frame

# Modified UVG dataset class that returns pairs of consecutive frames
class UVGPairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, precomputed_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing original UVG video frames
            precomputed_dir (string): Directory containing precomputed reference frames
            transform (callable, optional): Optional transform to be applied to frames
        """
        self.root_dir = root_dir
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        self.frame_pairs = []

        # UVG videos
        video_names = [
            'Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV', 
            'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV', 
            'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV', 
            'YachtRide_1920x1024_120fps_420_8bit_YUV'
        ]

        # Get consecutive frame pairs for each video
        for video_name in video_names:
            video_dir = os.path.join(root_dir, video_name)
            precomp_video_dir = os.path.join(self.precomputed_dir, video_name)
            
            if os.path.isdir(video_dir) and os.path.isdir(precomp_video_dir):
                # Get all original frames
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])
                
                # Create consecutive frame pairs
                for i in range(len(frames) - 1):
                    # Current frame is the next frame in the original sequence
                    curr_frame = frames[i+1]
                    # Reference frame is the precomputed version of the previous frame (with "ref_" prefix)
                    ref_frame = f"ref_{frames[i]}"
                    
                    # Ensure the reference frame file exists
                    if os.path.exists(os.path.join(precomp_video_dir, ref_frame)):
                        self.frame_pairs.append({
                            'video': video_name,
                            'curr_frame': curr_frame,
                            'ref_frame': ref_frame
                        })

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.frame_pairs[idx]
        video_name = pair['video']
        ref_frame_name = pair['ref_frame']
        curr_frame_name = pair['curr_frame']
        
        # Load reference frame from precomputed directory
        ref_path = os.path.join(self.precomputed_dir, video_name, ref_frame_name)
        ref_frame = Image.open(ref_path).convert('RGB')
        
        # Load current frame from original directory
        curr_path = os.path.join(self.root_dir, video_name, curr_frame_name)
        curr_frame = Image.open(curr_path).convert('RGB')

        # Apply transform if provided
        if self.transform:
            ref_frame = self.transform(ref_frame)
            curr_frame = self.transform(curr_frame)

        return ref_frame, curr_frame

# Modified training function without dual loops
def train_one_epoch(model, train_loader, optimizer, device, stage, epoch, gradient_accumulation_steps=1):
    """
    Train for one epoch with simplified processing of frame pairs.
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
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
    
    # Process batches of frame pairs
    progress_bar = tqdm(train_loader)
    
    for batch_idx, (reference_frames, current_frames) in enumerate(progress_bar):
        batch_size = reference_frames.size(0)
        
        # Zero gradients
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Move data to device
        reference_frames = reference_frames.to(device)
        current_frames = current_frames.to(device)
        
        # Process the batch of frame pairs
        result = model(reference_frames, current_frames, training=True, stage=stage)
        
        # Calculate loss and backpropagate
        loss = result["loss"] / gradient_accumulation_steps
        loss.backward()
        
        # Apply optimizer step after accumulating gradients
        optimizer.step()

        # Collect statistics
        total_loss += result["loss"].item() * batch_size
        total_mse += result["mse_loss"].item() * batch_size
        total_bpp += result["bpp_train"].item() * batch_size
        total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
        n_frames += batch_size
        
        # Update progress bar
        progress_bar.set_description(
            f"Epoch {epoch} Stage {stage} | "
            f"Loss: {total_loss / n_frames:.4f}, "
            f"MSE: {total_mse / n_frames:.6f}, "
            f"BPP: {total_bpp / n_frames:.4f}, "
            f"PSNR: {total_psnr / n_frames:.4f}"
        )
    
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

# Modified evaluation function without dual loops
def evaluate(model, test_loader, device, stage):
    """
    Evaluate model using simplified processing of frame pairs
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    with torch.no_grad():
        for reference_frames, current_frames in test_loader:
            batch_size = reference_frames.size(0)
            
            # Move data to device
            reference_frames = reference_frames.to(device)
            current_frames = current_frames.to(device)
            
            result = model(reference_frames, current_frames, training=False, stage=stage)
            
            # Collect statistics
            total_loss += result["loss"].item() * batch_size
            total_mse += result["mse_loss"].item() * batch_size
            total_bpp += result["bpp_train"].item() * batch_size
            total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
            
            n_frames += batch_size
    
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
    parser = argparse.ArgumentParser(description='DCVC Training with Enhanced Dataset')
    parser.add_argument('--vimeo_dir', type=str, required=True, help='Path to Vimeo-90k dataset')
    parser.add_argument('--precomputed_dir', type=str, required=True, help='Path to precomputed directory')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_data_pre_rs_sl', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs_data_pre_rs_sl', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (0-3) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4], help='Training stage (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='1', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='psnr', choices=['psnr', 'ms-ssim'],
                        help='Model type: psnr or ms-ssim')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--previous_stage_checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous stage to resume from')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    parser.add_argument('--precomputed_dir_uvg', type=str, required=True, help='Path to precomputed directory for UVG dataset')
    
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

    # Create training dataset with the new pair structure
    train_dataset = Vimeo90kPairDataset(
        root_dir=args.vimeo_dir,
        precomputed_dir=os.path.join(args.precomputed_dir, str(args.quality_index)),
        septuplet_list=args.septuplet_list,
        transform=transform,
        crop_size=args.crop_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create test dataset (UVG) with the new pair structure
    test_dataset = UVGPairDataset(
        root_dir=args.uvg_dir,
        precomputed_dir=os.path.join(args.precomputed_dir_uvg, str(args.quality_index)),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Make script print info about UVG dataset at start
    print(f"UVG dataset loaded with {len(test_dataset)} frame pairs.")

    print(
        f"Training model with lambda = {args.lambda_value}, quality_index = {args.quality_index}, stage = {args.stage}")

    # Initialize DCVC model
    model = DCVC_net(lmbda=args.lambda_value)
    model = model.to(device)

    # Compile the model if specified
    if args.compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # Initialize optimizer
    if args.stage == 4 and args.previous_stage_checkpoint:
        # Use lower learning rate for stage 4
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
            optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience, verbose=True
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

    # Initialize starting epoch and best loss
    start_epoch = 0
    best_loss = float('inf')

    # Log file
    stage_descriptions = {
        1: "Warm up MV generation part",
        2: "Train other modules",
        3: "Train with bit cost",
        4: "End-to-end training (using precomputed references)"
    }

    log_file = os.path.join(args.log_dir,
                            f'train_log_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_{args.model_type}.txt')

    # Check for SpyNet initialization
    if args.spynet_checkpoint:
        print(f"Initializing motion estimation network with pretrained SpyNet weights: {args.spynet_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_checkpoint, map_location=device)
        
        spynet_state_dict = spynet_checkpoint['state_dict']
        unwrap_model(model).opticFlow.load_state_dict(spynet_state_dict, strict=True)
        print("Loaded SpyNet weights directly into opticFlow component")
    
    if args.spynet_from_dcvc_checkpoint:
        print(f"Initializing motion estimation network with SpyNet weights from DCVC checkpoint: {args.spynet_from_dcvc_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_from_dcvc_checkpoint, map_location=device)

        # Extract only SpyNet weights
        spynet_state_dict = {}
        for key, value in spynet_checkpoint.items():
            if key.startswith('opticFlow.'):
                spynet_state_dict[key.replace('opticFlow.', '')] = value
        
        unwrap_model(model).opticFlow.load_state_dict(spynet_state_dict, strict=True)
        print("Loaded SpyNet weights from DCVC checkpoint into opticFlow component")

    # Resume training from checkpoint if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if this is a state_dict only or a complete checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_loss = checkpoint['best_loss']
                
                # Load scheduler state if it exists and scheduler is initialized
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.6f}")
            else:
                # State dict only
                unwrap_model(model).load_dict(checkpoint)
                print("Loaded model weights only (no training state)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    # Load from previous stage checkpoint if no resume but previous_stage_checkpoint is specified
    elif args.previous_stage_checkpoint:
        print(f"Loading model from previous stage checkpoint: {args.previous_stage_checkpoint}")
        try:
            checkpoint = torch.load(args.previous_stage_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_dict(checkpoint['model_state_dict'])
                print("Loaded model weights only (no training state)")
            else:
                unwrap_model(model).load_dict(checkpoint)  # Use load_dict method as defined in DCVC_net
            print("Successfully loaded model from previous stage")
            # Test result instantly for the previous stage
            test_stats = evaluate(
                model, test_loader, device, args.stage-1
            )
            # Log results
            with open(log_file, 'a') as f:
                f.write(f"From stage {args.stage-1}:\n")
                f.write(f"  Test Loss: {test_stats['loss']:.6f}\n")
                f.write(f"  Test MSE: {test_stats['mse']:.6f}\n")
                f.write(f"  Test PSNR: {test_stats['psnr']:.4f}\n")
                f.write(f"  Test BPP: {test_stats['bpp']:.6f}\n")

        except Exception as e:
            print(f"Error loading previous stage checkpoint: {e}")
            print("Starting training from scratch")

    with open(log_file, 'a') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lambda value: {args.lambda_value}\n")
        f.write(f"Quality index: {args.quality_index}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Stage: {args.stage} ({stage_descriptions[args.stage]})\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"LR Scheduler: {args.lr_scheduler}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
        f.write(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}\n")
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
            f.write(f"SpyNet initialization from DCVC checkpoint: {args.spynet_from_dcvc_checkpoint}\n")
        f.write(f"Training dataset: {len(train_dataset)} frame pairs\n")
        f.write(f"UVG dataset: {len(test_dataset)} frame pairs\n")
        f.write(f"Using simplified frame pair processing\n")
        f.write("=" * 80 + "\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}\n")

        # Apply warmup scheduler if in warmup phase
        if warmup_scheduler is not None and epoch < args.lr_warmup_epochs:
            warmup_scheduler.step()
        
        # Train one epoch with simplified frame pair processing
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            args.stage, epoch + 1,
            args.gradient_accumulation_steps
        )

        # Evaluate on test set
        test_stats = evaluate(
            model, test_loader, device, args.stage
        )

        # Step scheduler after training (different for ReduceLROnPlateau)
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(test_stats['loss'])
            elif epoch >= args.lr_warmup_epochs:  # Only step main scheduler after warmup
                scheduler.step()

        # Log results
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
            f.write("=" * 80 + "\n")

        # Save latest checkpoint with training state for resuming
        latest_checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest.pth'
        )
        
        # Create save dictionary with all training state
        save_dict = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(model).state_dict(),
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
        
        # Save the checkpoint
        torch.save(save_dict, latest_checkpoint_path)

        # Save best checkpoint if current test loss is the best so far
        if test_stats['loss'] < best_loss:
            best_loss = test_stats['loss']
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best.pth'
            )
            # Save full training state for the best model too
            torch.save(save_dict, best_checkpoint_path)
            print(f"New best model saved with test loss: {best_loss:.6f}")
            # Write to log
            with open(log_file, 'a') as f:
                f.write(f"New best model saved with test loss: {best_loss:.6f}\n")

        print(f"Epoch {epoch + 1}/{args.epochs} completed. Latest checkpoint saved.")

    # Save final model for this stage (state_dict only for compatibility with original code)
    final_model_path = os.path.join(
        args.checkpoint_dir,
        f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}.pth'
    )
    torch.save(unwrap_model(model).state_dict(), final_model_path)
    print(f"Final model for stage {args.stage} saved to {final_model_path}")

    # If this is the final stage (4), also save with the standard naming convention
    if args.stage == 4:
        standard_model_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_quality_{args.quality_index}_{args.model_type}.pth'
        )
        torch.save(unwrap_model(model).state_dict(), standard_model_path)
        print(f"Final model (standard name) saved to {standard_model_path}")

    with open(log_file, 'a') as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    print(f"Training completed for stage {args.stage}!")


if __name__ == '__main__':
    main()