import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import torchvision.transforms as transforms
import time
import json
from timm.utils import unwrap_model

# Import the model architectures
from src.zoo.image import model_architectures as architectures

# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a dataset class for UVG that returns frames to be compressed
class UVGFramesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with UVG videos.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_paths = []
        
        # UVG videos
        video_names = [
            'Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV', 
            'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV', 
            'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV', 
            'YachtRide_1920x1024_120fps_420_8bit_YUV'
        ]
        
        # Collect all frame paths from each video
        for video_name in video_names:
            video_dir = os.path.join(root_dir, video_name)
            if os.path.isdir(video_dir):
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])
                for frame in frames:
                    self.frame_paths.append({
                        'video': video_name,
                        'frame': frame,
                        'path': os.path.join(video_dir, frame)
                    })

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_info = self.frame_paths[idx]
        image = Image.open(frame_info['path']).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, frame_info  # Return frame and its metadata

def precompute_uvg_reference_frames(dataset, i_frame_model, device, save_dir, quality_index, batch_size=16, num_workers=4):
    """
    Precompute compressed reference frames for UVG dataset using I-frame model
    
    Args:
        dataset: Dataset containing original frames
        i_frame_model: Pretrained I-frame model
        device: Device to run I-frame model on
        save_dir: Base directory to save compressed frames
        quality_index: Quality index (0-3) to organize the saved frames
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
    """
    # Create dataloader for efficient processing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set model to eval mode
    i_frame_model.eval()
    
    # Create the quality-specific save directory
    quality_save_dir = os.path.join(save_dir, str(quality_index))
    os.makedirs(quality_save_dir, exist_ok=True)
    
    # Create metadata to store precomputed frame info
    metadata = {
        'frame_count': 0,
        'quality_index': quality_index,
        'videos': {}
    }
    
    # Process all frames
    with torch.no_grad():
        for batch_idx, (batch_frames, batch_info) in enumerate(tqdm(dataloader, desc=f"Compressing UVG frames (Quality {quality_index})")):
            # Move batch to device
            batch_frames = batch_frames.to(device)
            
            # Compress frames using I-frame model
            results = i_frame_model(batch_frames)
            compressed_frames = results["x_hat"].cpu()
            
            # Save each compressed frame
            for i in range(len(batch_frames)):
                video_name = batch_info['video'][i]
                frame_name = batch_info['frame'][i]
                
                # Create directory for this video
                video_save_dir = os.path.join(quality_save_dir, video_name)
                os.makedirs(video_save_dir, exist_ok=True)
                
                # Track video in metadata
                if video_name not in metadata['videos']:
                    metadata['videos'][video_name] = []
                
                # Save compressed frame
                frame_path = os.path.join(video_save_dir, f"ref_{frame_name}")
                frame_tensor = compressed_frames[i]
                
                # 关键修复: 在保存前裁剪像素值到有效范围 [0, 1]
                frame_tensor = frame_tensor.clamp(0, 1)
                
                # Convert tensor to PIL image and save as uncompressed PNG
                frame_image = transforms.ToPILImage()(frame_tensor)
                frame_image.save(frame_path, format='PNG', compress_level=0)
                
                # Update metadata
                metadata['frame_count'] += 1
                metadata['videos'][video_name].append(frame_name)
    
    # Save metadata for this quality level
    with open(os.path.join(quality_save_dir, 'uvg_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Precomputed {metadata['frame_count']} UVG reference frames for quality {quality_index}, saved to {quality_save_dir}")
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Precompute Reference Frames for UVG Test Dataset')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    parser.add_argument('--save_dir', type=str, required=True, help='Base directory to save precomputed reference frames')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', help='I-frame model name')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--quality_index', type=int, required=True, choices=[0, 1, 2, 3], help='Quality index (0-3)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset for UVG frames to compress
    dataset = UVGFramesDataset(
        root_dir=args.uvg_dir,
        transform=transform
    )

    print(f"UVG dataset loaded with {len(dataset)} frames.")

    # Load I-frame model
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_load_checkpoint).eval()
    i_frame_model = i_frame_model.to(device)
    
    # Compile model if specified
    if args.compile:
        print("Compiling the I-frame model...")
        i_frame_model = torch.compile(i_frame_model)

    # Precompute and save reference frames
    start_time = time.time()
    metadata = precompute_uvg_reference_frames(
        dataset=dataset,
        i_frame_model=i_frame_model,
        device=device,
        save_dir=args.save_dir,
        quality_index=args.quality_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    elapsed_time = time.time() - start_time
    
    print(f"UVG precomputation completed in {elapsed_time:.2f} seconds")
    print(f"Processed {metadata['frame_count']} frames from {len(metadata['videos'])} videos")
    print(f"Results saved to {args.save_dir}/{args.quality_index}")

if __name__ == '__main__':
    main()
    # Example usage:
    # python preprocess_test.py --uvg_dir /home/yichi/Project/dataset/UVG/png_sequences --save_dir /home/yichi/Project/dataset/UVG/reference_sequences --i_frame_model_path /home/yichi/Project/OpenDCVC/DCVC-family/DCVC/checkpoints/cheng2020-anchor-3-e49be189.pth.tar --quality_index 0 --cuda_device 0
    # python preprocess_test.py --uvg_dir /home/yichi/Project/dataset/UVG/png_sequences --save_dir /home/yichi/Project/dataset/UVG/reference_sequences --i_frame_model_path /home/yichi/Project/OpenDCVC/DCVC-family/DCVC/checkpoints/cheng2020-anchor-4-98b0b468.pth.tar --quality_index 1 --cuda_device 1
    # python preprocess_test.py --uvg_dir /home/yichi/Project/dataset/UVG/png_sequences --save_dir /home/yichi/Project/dataset/UVG/reference_sequences --i_frame_model_path /home/yichi/Project/OpenDCVC/DCVC-family/DCVC/checkpoints/cheng2020-anchor-5-23852949.pth.tar --quality_index 2 --cuda_device 0
    # python preprocess_test.py --uvg_dir /home/yichi/Project/dataset/UVG/png_sequences --save_dir /home/yichi/Project/dataset/UVG/reference_sequences --i_frame_model_path /home/yichi/Project/OpenDCVC/DCVC-family/DCVC/checkpoints/cheng2020-anchor-6-4c052b1a.pth.tar --quality_index 3 --cuda_device 1