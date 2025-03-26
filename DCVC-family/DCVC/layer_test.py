import torch
import time
import numpy as np
from tabulate import tabulate

# Importing your model
# Assuming the model is in a package structure like:
# from dcvc.models.video_model import DCVC_net
# Adjust these imports based on your actual package structure
# For this test, let's assume DCVC_net is already defined
from src.models.DCVC_net import DCVC_net
def test_layer_forward_time(model, reference_frame, input_frame, num_runs=10):
    """
    Test the forward propagation time for each layer in the DCVC_net model.
    
    Args:
        model (DCVC_net): The model to test
        reference_frame (torch.Tensor): Reference frame tensor of shape [N, C, H, W]
        input_frame (torch.Tensor): Input frame tensor of shape [N, C, H, W]
        num_runs (int): Number of runs to average timing results
    
    Returns:
        dict: Dictionary containing timing results for each layer
    """
    device = next(model.parameters()).device
    results = {}
    
    # Move data to the same device as model
    reference_frame = reference_frame.to(device)
    input_frame = input_frame.to(device)
    
    # Function to time a layer's forward pass
    def time_layer(name, forward_func, *args):
        # Warm-up run
        _ = forward_func(*args)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = forward_func(*args)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[name] = {"avg_time": avg_time, "std_time": std_time}
        return avg_time
    
    print(f"Testing forward time for each layer in DCVC_net (averaged over {num_runs} runs)...")
    
    # Optical Flow
    time_layer("opticFlow", model.opticFlow, input_frame, reference_frame)
    
    # Motion Vector Encoding
    estmv = model.opticFlow(input_frame, reference_frame)
    time_layer("mvEncoder", model.mvEncoder, estmv)
    
    # Motion Vector Prior Encoding
    mvfeature = model.mvEncoder(estmv)
    time_layer("mvpriorEncoder", model.mvpriorEncoder, mvfeature)
    
    # Motion Vector Prior Decoding
    z_mv = model.mvpriorEncoder(mvfeature)
    compressed_z_mv = torch.round(z_mv)
    time_layer("mvpriorDecoder", model.mvpriorDecoder, compressed_z_mv)
    
    # Auto-regressive Motion Vector
    quant_mv = torch.round(mvfeature)
    time_layer("auto_regressive_mv", model.auto_regressive_mv, quant_mv)
    
    # Motion Vector Entropy Parameters
    params_mv = model.mvpriorDecoder(compressed_z_mv)
    ctx_params_mv = model.auto_regressive_mv(quant_mv)
    time_layer("entropy_parameters_mv", model.entropy_parameters_mv, 
              torch.cat((params_mv, ctx_params_mv), dim=1))
    
    # Motion Vector Decoder Part 1
    time_layer("mvDecoder_part1", model.mvDecoder_part1, quant_mv)
    
    # Motion Vector Refinement
    quant_mv_upsample = model.mvDecoder_part1(quant_mv)
    time_layer("mv_refine", model.mv_refine, reference_frame, quant_mv_upsample)
    
    # Motion Compensation
    quant_mv_upsample_refine = model.mv_refine(reference_frame, quant_mv_upsample)
    time_layer("motioncompensation", model.motioncompensation, reference_frame, quant_mv_upsample_refine)
    
    # Feature Extraction and Context Refinement
    time_layer("feature_extract", model.feature_extract, reference_frame)
    
    ref_feature = model.feature_extract(reference_frame)
    prediction_init = torch.nn.functional.grid_sample(
        ref_feature, 
        quant_mv_upsample_refine.permute(0, 2, 3, 1), 
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )
    time_layer("context_refine", model.context_refine, prediction_init)
    
    # Temporal Prior Encoding
    context = model.motioncompensation(reference_frame, quant_mv_upsample_refine)
    time_layer("temporalPriorEncoder", model.temporalPriorEncoder, context)
    
    # Contextual Encoding
    time_layer("contextualEncoder", model.contextualEncoder, 
              torch.cat((input_frame, context), dim=1))
    
    # Prior Encoding/Decoding
    feature = model.contextualEncoder(torch.cat((input_frame, context), dim=1))
    time_layer("priorEncoder", model.priorEncoder, feature)
    
    z = model.priorEncoder(feature)
    compressed_z = torch.round(z)
    time_layer("priorDecoder", model.priorDecoder, compressed_z)
    
    # Auto-regressive and Entropy Parameters
    compressed_y_renorm = torch.round(feature)
    time_layer("auto_regressive", model.auto_regressive, compressed_y_renorm)
    
    params = model.priorDecoder(compressed_z)
    temporal_prior_params = model.temporalPriorEncoder(context)
    ctx_params = model.auto_regressive(compressed_y_renorm)
    time_layer("entropy_parameters", model.entropy_parameters,
              torch.cat((temporal_prior_params, params, ctx_params), dim=1))
    
    # Contextual Decoding
    time_layer("contextualDecoder_part1", model.contextualDecoder_part1, compressed_y_renorm)
    
    recon_image_feature = model.contextualDecoder_part1(compressed_y_renorm)
    time_layer("contextualDecoder_part2", model.contextualDecoder_part2, 
              torch.cat((recon_image_feature, context), dim=1))
    
    # Bit Estimation
    gaussian_params = model.entropy_parameters(
        torch.cat((temporal_prior_params, params, ctx_params), dim=1)
    )
    means_hat, scales_hat = gaussian_params.chunk(2, 1)
    scales_hat = torch.nn.functional.softplus(scales_hat+2.3)-2.3
    scales_hat = torch.exp(scales_hat)
    
    time_layer("feature_probs_based_sigma", model.feature_probs_based_sigma, 
              feature, means_hat, scales_hat, True)
    
    time_layer("iclr18_estrate_bits_z", model.iclr18_estrate_bits_z, compressed_z)
    time_layer("iclr18_estrate_bits_z_mv", model.iclr18_estrate_bits_z_mv, compressed_z_mv)
    
    # Full Model Forward Pass
    time_layer("full_forward", model, reference_frame, input_frame, True, 4)  # Stage 4 for full training
    
    return results

def create_random_frames(batch_size=1, channels=3, height=256, width=256, device='cuda'):
    """Create random frame tensors for testing"""
    ref_frame = torch.rand(batch_size, channels, height, width, device=device)
    input_frame = torch.rand(batch_size, channels, height, width, device=device)
    return ref_frame, input_frame

def display_results(results):
    """Display timing results in a formatted table"""
    # Sort results by average time (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'], reverse=True)
    
    # Prepare table data
    table_data = []
    for layer_name, timing in sorted_results:
        avg_ms = timing['avg_time'] * 1000  # Convert to milliseconds
        std_ms = timing['std_time'] * 1000  # Convert to milliseconds
        percentage = (avg_ms / results['full_forward']['avg_time'] / 1000) * 100
        table_data.append([layer_name, f"{avg_ms:.2f} ± {std_ms:.2f}", f"{percentage:.2f}%"])
    
    # Print the table
    headers = ["Layer", "Time (ms)", "% of Total"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print total time
    total_time = results['full_forward']['avg_time'] * 1000
    print(f"\nTotal forward pass time: {total_time:.2f} ms")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model (adjust lambda as needed)
    model = DCVC_net(lmbda=1.0).to(device)
    model.eval()  # Set to evaluation mode
    
    # Create random input frames
    ref_frame, input_frame = create_random_frames(device=device)
    
    # Test forward time
    with torch.no_grad():  # Disable gradient computation for timing
        results = test_layer_forward_time(model, ref_frame, input_frame)
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=6 python layer_test.py