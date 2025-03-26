import torch
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch.autograd.profiler as profiler
import os
import json

# Import your model - adjust this import based on your actual package structure
# from dcvc.models.video_model import DCVC_net
from src.models.DCVC_net import DCVC_net
class DCVCProfiler:
    """Class for comprehensive profiling of DCVC network performance"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def create_random_frames(self, batch_size=1, channels=3, height=256, width=256):
        """Create random frame tensors for testing"""
        ref_frame = torch.rand(batch_size, channels, height, width, device=self.device)
        input_frame = torch.rand(batch_size, channels, height, width, device=self.device)
        return ref_frame, input_frame
    
    def profile_layer_timing(self, ref_frame, input_frame, num_runs=10, warmup=3):
        """Profile the timing of each layer in the model"""
        results = {}
        
        # Define all layers to test
        layers_to_test = {
            "opticFlow": lambda: self.model.opticFlow(input_frame, ref_frame),
            "feature_extract": lambda: self.model.feature_extract(ref_frame),
            "context_refine": lambda: self.model.context_refine(self.model.feature_extract(ref_frame)),
            "mvEncoder": lambda: self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)),
            "mvpriorEncoder": lambda: self.model.mvpriorEncoder(
                self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame))
            ),
            "mvpriorDecoder": lambda: self.model.mvpriorDecoder(
                torch.round(self.model.mvpriorEncoder(
                    self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame))
                ))
            ),
            "mvDecoder_part1": lambda: self.model.mvDecoder_part1(
                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
            ),
            "mv_refine": lambda: self.model.mv_refine(
                ref_frame, 
                self.model.mvDecoder_part1(
                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                )
            ),
            "motioncompensation": lambda: self.model.motioncompensation(
                ref_frame,
                self.model.mv_refine(
                    ref_frame, 
                    self.model.mvDecoder_part1(
                        torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                    )
                )
            ),
            "temporalPriorEncoder": lambda: self.model.temporalPriorEncoder(
                self.model.motioncompensation(
                    ref_frame,
                    self.model.mv_refine(
                        ref_frame, 
                        self.model.mvDecoder_part1(
                            torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                        )
                    )
                )
            ),
            "contextualEncoder": lambda: self.model.contextualEncoder(
                torch.cat((
                    input_frame, 
                    self.model.motioncompensation(
                        ref_frame,
                        self.model.mv_refine(
                            ref_frame, 
                            self.model.mvDecoder_part1(
                                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                            )
                        )
                    )
                ), dim=1)
            ),
            "priorEncoder": lambda: self.model.priorEncoder(
                self.model.contextualEncoder(
                    torch.cat((
                        input_frame, 
                        self.model.motioncompensation(
                            ref_frame,
                            self.model.mv_refine(
                                ref_frame, 
                                self.model.mvDecoder_part1(
                                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                )
                            )
                        )
                    ), dim=1)
                )
            ),
            "priorDecoder": lambda: self.model.priorDecoder(
                torch.round(self.model.priorEncoder(
                    self.model.contextualEncoder(
                        torch.cat((
                            input_frame, 
                            self.model.motioncompensation(
                                ref_frame,
                                self.model.mv_refine(
                                    ref_frame, 
                                    self.model.mvDecoder_part1(
                                        torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                    )
                                )
                            )
                        ), dim=1)
                    )
                ))
            ),
            "auto_regressive": lambda: self.model.auto_regressive(
                torch.round(self.model.contextualEncoder(
                    torch.cat((
                        input_frame, 
                        self.model.motioncompensation(
                            ref_frame,
                            self.model.mv_refine(
                                ref_frame, 
                                self.model.mvDecoder_part1(
                                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                )
                            )
                        )
                    ), dim=1)
                ))
            ),
            "auto_regressive_mv": lambda: self.model.auto_regressive_mv(
                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
            ),
            "entropy_parameters": lambda: self.model.entropy_parameters(
                torch.cat((
                    self.model.temporalPriorEncoder(
                        self.model.motioncompensation(
                            ref_frame,
                            self.model.mv_refine(
                                ref_frame, 
                                self.model.mvDecoder_part1(
                                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                )
                            )
                        )
                    ),
                    self.model.priorDecoder(
                        torch.round(self.model.priorEncoder(
                            self.model.contextualEncoder(
                                torch.cat((
                                    input_frame, 
                                    self.model.motioncompensation(
                                        ref_frame,
                                        self.model.mv_refine(
                                            ref_frame, 
                                            self.model.mvDecoder_part1(
                                                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                            )
                                        )
                                    )
                                ), dim=1)
                            )
                        ))
                    ),
                    self.model.auto_regressive(
                        torch.round(self.model.contextualEncoder(
                            torch.cat((
                                input_frame, 
                                self.model.motioncompensation(
                                    ref_frame,
                                    self.model.mv_refine(
                                        ref_frame, 
                                        self.model.mvDecoder_part1(
                                            torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                        )
                                    )
                                )
                            ), dim=1)
                        ))
                    )
                ), dim=1)
            ),
            "entropy_parameters_mv": lambda: self.model.entropy_parameters_mv(
                torch.cat((
                    self.model.mvpriorDecoder(
                        torch.round(self.model.mvpriorEncoder(
                            self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame))
                        ))
                    ),
                    self.model.auto_regressive_mv(
                        torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                    )
                ), dim=1)
            ),
            "contextualDecoder_part1": lambda: self.model.contextualDecoder_part1(
                torch.round(self.model.contextualEncoder(
                    torch.cat((
                        input_frame, 
                        self.model.motioncompensation(
                            ref_frame,
                            self.model.mv_refine(
                                ref_frame, 
                                self.model.mvDecoder_part1(
                                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                )
                            )
                        )
                    ), dim=1)
                ))
            ),
            "contextualDecoder_part2": lambda: self.model.contextualDecoder_part2(
                torch.cat((
                    self.model.contextualDecoder_part1(
                        torch.round(self.model.contextualEncoder(
                            torch.cat((
                                input_frame, 
                                self.model.motioncompensation(
                                    ref_frame,
                                    self.model.mv_refine(
                                        ref_frame, 
                                        self.model.mvDecoder_part1(
                                            torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                        )
                                    )
                                )
                            ), dim=1)
                        ))
                    ),
                    self.model.motioncompensation(
                        ref_frame,
                        self.model.mv_refine(
                            ref_frame, 
                            self.model.mvDecoder_part1(
                                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                            )
                        )
                    )
                ), dim=1)
            ),
            "full_forward": lambda: self.model(ref_frame, input_frame, training=False, stage=4)
        }
        
        print(f"Profiling layer timing (average over {num_runs} runs with {warmup} warmup runs)...")
        
        # Run timing tests
        for layer_name, layer_func in layers_to_test.items():
            print(f"Testing {layer_name}...", end="", flush=True)
            
            # Warmup runs
            for _ in range(warmup):
                _ = layer_func()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
            # Timed runs
            times = []
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = layer_func()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[layer_name] = {"avg_time": avg_time, "std_time": std_time}
            print(f" {avg_time*1000:.2f} ms")
            
        return results
    
    def profile_memory_usage(self, ref_frame, input_frame):
        """Profile memory usage of each layer in the model"""
        memory_usage = {}
        
        # Define a helper function to measure peak memory
        def measure_peak_memory(func):
            if self.device.type != 'cuda':
                return 0  # Memory tracking only works on CUDA
            
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            _ = func()
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() - start_mem
            return peak_mem
        
        # Test same layers as in timing
        layers_to_test = {
            "opticFlow": lambda: self.model.opticFlow(input_frame, ref_frame),
            "mvEncoder": lambda: self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)),
            "contextualEncoder": lambda: self.model.contextualEncoder(
                torch.cat((
                    input_frame, 
                    self.model.motioncompensation(
                        ref_frame,
                        self.model.mv_refine(
                            ref_frame, 
                            self.model.mvDecoder_part1(
                                torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                            )
                        )
                    )
                ), dim=1)
            ),
            "contextualDecoder_part1": lambda: self.model.contextualDecoder_part1(
                torch.round(self.model.contextualEncoder(
                    torch.cat((
                        input_frame, 
                        self.model.motioncompensation(
                            ref_frame,
                            self.model.mv_refine(
                                ref_frame, 
                                self.model.mvDecoder_part1(
                                    torch.round(self.model.mvEncoder(self.model.opticFlow(input_frame, ref_frame)))
                                )
                            )
                        )
                    ), dim=1)
                ))
            ),
            "full_forward": lambda: self.model(ref_frame, input_frame, training=False, stage=4)
        }
        
        print("Profiling memory usage...")
        
        for layer_name, layer_func in layers_to_test.items():
            print(f"Testing {layer_name}...", end="", flush=True)
            mem = measure_peak_memory(layer_func)
            memory_usage[layer_name] = mem / (1024 * 1024)  # Convert to MB
            print(f" {memory_usage[layer_name]:.2f} MB")
            
        return memory_usage
    
    def run_detailed_profiler(self, ref_frame, input_frame):
        """Run PyTorch profiler for detailed operation breakdown"""
        print("Running detailed profiler...")
        
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            _ = self.model(ref_frame, input_frame, training=False, stage=4)
            
        return prof
    
    def profile_resolution_scaling(self, resolutions=[(256, 256), (512, 512), (1024, 1024), (1920, 1024)]):
        """Test how model performance scales with different input resolutions"""
        scaling_results = {}
        
        print("Testing resolution scaling...")
        
        for height, width in resolutions:
            print(f"Testing {width}x{height}...")
            ref_frame, input_frame = self.create_random_frames(
                batch_size=1, channels=3, height=height, width=width
            )
            
            # Measure time
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = self.model(ref_frame, input_frame, training=False, stage=4)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Measure memory if CUDA
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                _ = self.model(ref_frame, input_frame, training=False, stage=4)
                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory = 0
                
            scaling_results[(height, width)] = {
                "time": end_time - start_time,
                "memory": memory,
                "pixels": height * width
            }
            
            print(f"  Time: {(end_time - start_time)*1000:.2f} ms, Memory: {memory:.2f} MB")
            
        return scaling_results
    
    def profile_batch_scaling(self, max_batch=8, frame_size=(256, 256)):
        """Test how model performance scales with different batch sizes"""
        batch_results = {}
        height, width = frame_size
        batch_sizes = [1, 2, 4, 8, 16] if max_batch >= 16 else [1, 2, 4, 8][:max_batch]
        
        print("Testing batch size scaling...")
        
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch size {batch_size}...")
                ref_frame, input_frame = self.create_random_frames(
                    batch_size=batch_size, channels=3, height=height, width=width
                )
                
                # Measure time
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = self.model(ref_frame, input_frame, training=False, stage=4)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                # Measure memory if CUDA
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    _ = self.model(ref_frame, input_frame, training=False, stage=4)
                    torch.cuda.synchronize()
                    memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                else:
                    memory = 0
                    
                batch_results[batch_size] = {
                    "time": end_time - start_time,
                    "memory": memory,
                    "time_per_frame": (end_time - start_time) / batch_size
                }
                
                print(f"  Time: {(end_time - start_time)*1000:.2f} ms, Memory: {memory:.2f} MB")
                
            except RuntimeError as e:
                print(f"  Failed with batch size {batch_size}: {str(e)}")
                break
                
        return batch_results
    
    def display_layer_timing_results(self, results):
        """Display layer timing results in a formatted table"""
        # Sort by time (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'], reverse=True)
        
        # Prepare data for table
        table_data = []
        total_time = results['full_forward']['avg_time'] * 1000  # ms
        
        for layer_name, timing in sorted_results:
            avg_ms = timing['avg_time'] * 1000
            std_ms = timing['std_time'] * 1000
            percentage = (avg_ms / total_time) * 100
            table_data.append([layer_name, f"{avg_ms:.2f} ± {std_ms:.2f}", f"{percentage:.2f}%"])
            
        # Print formatted table
        headers = ["Layer", "Time (ms)", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal forward pass time: {total_time:.2f} ms")
        
    def visualize_timing_results(self, results, output_path="layer_timing.png"):
        """Visualize layer timing results as a horizontal bar chart"""
        # Sort by time (ascending for better visualization)
        sorted_results = sorted(
            [(k, v['avg_time'] * 1000) for k, v in results.items() if k != 'full_forward'], 
            key=lambda x: x[1]
        )
        
        layers = [item[0] for item in sorted_results]
        times = [item[1] for item in sorted_results]
        
        plt.figure(figsize=(12, 10))
        plt.barh(layers, times, color='skyblue')
        plt.xlabel('Time (ms)')
        plt.title('Layer-wise Forward Pass Timing')
        plt.tight_layout()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        
    def visualize_resolution_scaling(self, scaling_results, output_path="resolution_scaling.png"):
        """Visualize how performance scales with resolution"""
        resolutions = [f"{w}x{h}" for h, w in scaling_results.keys()]
        times = [res['time'] * 1000 for res in scaling_results.values()]  # ms
        pixels = [res['pixels'] for res in scaling_results.values()]  # total pixels
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Resolution')
        ax1.set_ylabel('Processing Time (ms)', color=color)
        ax1.bar(resolutions, times, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add second y-axis for memory
        if scaling_results[list(scaling_results.keys())[0]]['memory'] > 0:
            ax2 = ax1.twinx()
            color = 'tab:red'
            memories = [res['memory'] for res in scaling_results.values()]
            ax2.set_ylabel('Memory Usage (MB)', color=color)
            ax2.plot(resolutions, memories, 'o-', color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Performance Scaling with Resolution')
        plt.tight_layout()
        plt.savefig(output_path)
        
        # Create a second plot for time vs pixels
        plt.figure(figsize=(8, 6))
        plt.scatter(pixels, times)
        plt.plot(np.unique(pixels), np.poly1d(np.polyfit(pixels, times, 1))(np.unique(pixels)), color='red')
        plt.xlabel('Number of Pixels')
        plt.ylabel('Processing Time (ms)')
        plt.title('Processing Time vs. Pixel Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_path.replace('.png', '_pixels.png'))
        
    def visualize_batch_scaling(self, batch_results, output_path="batch_scaling.png"):
        """Visualize how performance scales with batch size"""
        batch_sizes = list(batch_results.keys())
        times = [res['time'] * 1000 for res in batch_results.values()]  # ms
        time_per_frame = [res['time_per_frame'] * 1000 for res in batch_results.values()]  # ms
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Total Processing Time (ms)', color=color)
        ax1.plot(batch_sizes, times, 'o-', color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Time per Frame (ms)', color=color)
        ax2.plot(batch_sizes, time_per_frame, 'o-', color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add memory if available
        if batch_results[batch_sizes[0]]['memory'] > 0:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))
            color = 'tab:red'
            memories = [res['memory'] for res in batch_results.values()]
            ax3.set_ylabel('Memory Usage (MB)', color=color)
            ax3.plot(batch_sizes, memories, 'o-', color=color, linewidth=2)
            ax3.tick_params(axis='y', labelcolor=color)
        
        plt.title('Performance Scaling with Batch Size')
        plt.tight_layout()
        plt.savefig(output_path)
        
    def profile_model_parameters(self):
        """Profile the number of parameters in each layer and module"""
        parameter_count = {}
        total_params = 0
        
        # Get parameter count for the whole model
        for name, param in self.model.named_parameters():
            module_name = name.split('.')[0]  # Get top-level module name
            param_count = param.numel()
            
            if module_name in parameter_count:
                parameter_count[module_name] += param_count
            else:
                parameter_count[module_name] = param_count
                
            total_params += param_count
        
        # Sort by parameter count (descending)
        sorted_params = sorted(parameter_count.items(), key=lambda x: x[1], reverse=True)
        
        # Print results
        print(f"Model Parameter Statistics:")
        print(f"Total parameters: {total_params:,}")
        print("\nParameter distribution by module:")
        
        table_data = []
        for module_name, count in sorted_params:
            percentage = (count / total_params) * 100
            table_data.append([module_name, f"{count:,}", f"{percentage:.2f}%"])
            
        headers = ["Module", "Parameters", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return parameter_count, total_params
        
    def run_comprehensive_profile(self, output_dir="profiling_results"):
        """Run a comprehensive profile and save all results"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate standard test frames
        ref_frame, input_frame = self.create_random_frames(batch_size=1, channels=3, height=256, width=256)
        
        # Profile layer timing
        print("\n=== Layer Timing Analysis ===")
        timing_results = self.profile_layer_timing(ref_frame, input_frame)
        self.display_layer_timing_results(timing_results)
        self.visualize_timing_results(timing_results, os.path.join(output_dir, "layer_timing.png"))
        
        # Memory usage
        if self.device.type == 'cuda':
            print("\n=== Memory Usage Analysis ===")
            memory_results = self.profile_memory_usage(ref_frame, input_frame)
        
        # Model parameters
        print("\n=== Model Parameter Analysis ===")
        param_count, total_params = self.profile_model_parameters()
        
        # Resolution scaling
        print("\n=== Resolution Scaling Analysis ===")
        resolution_results = self.profile_resolution_scaling()
        self.visualize_resolution_scaling(resolution_results, os.path.join(output_dir, "resolution_scaling.png"))
        
        # Batch scaling
        print("\n=== Batch Size Scaling Analysis ===")
        batch_results = self.profile_batch_scaling()
        self.visualize_batch_scaling(batch_results, os.path.join(output_dir, "batch_scaling.png"))
        
        # Detailed profiler (optional, can be very verbose)
        # prof = self.run_detailed_profiler(ref_frame, input_frame)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # prof.export_chrome_trace(os.path.join(output_dir, "trace.json"))
        
        # Save all results to JSON
        all_results = {
            "timing": {k: {"avg_time": float(v["avg_time"]), "std_time": float(v["std_time"])} 
                      for k, v in timing_results.items()},
            "parameters": {k: int(v) for k, v in param_count.items()},
            "total_parameters": int(total_params),
            "resolution_scaling": {f"{h}x{w}": {
                "time": float(v["time"]), 
                "memory": float(v["memory"]),
                "pixels": int(v["pixels"])
            } for (h, w), v in resolution_results.items()},
            "batch_scaling": {str(k): {
                "time": float(v["time"]), 
                "memory": float(v["memory"]),
                "time_per_frame": float(v["time_per_frame"])
            } for k, v in batch_results.items()}
        }
        
        if self.device.type == 'cuda':
            all_results["memory"] = {k: float(v) for k, v in memory_results.items()}
            
        with open(os.path.join(output_dir, "profiling_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nAll profiling results saved to {output_dir}/")
        return all_results


def main():
    """Main function to run the profiler"""
    parser = argparse.ArgumentParser(description="DCVC Network Performance Profiler")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lmbda",
                        help="Lambda value for rate-distortion tradeoff")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the profiling on (cuda or cpu)")
    parser.add_argument("--output", type=str, default="profiling_results",
                        help="Output directory for profiling results")
    parser.add_argument("--batch-test", action="store_true",
                        help="Run batch size scaling test")
    parser.add_argument("--res-test", action="store_true",
                        help="Run resolution scaling test")
    parser.add_argument("--layer-test", action="store_true",
                        help="Run layer-wise timing test")
    parser.add_argument("--param-test", action="store_true",
                        help="Run parameter count test")
    parser.add_argument("--memory-test", action="store_true",
                        help="Run memory usage test")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    
    args = parser.parse_args()
    
    # Check if at least one test is specified
    if not (args.batch_test or args.res_test or args.layer_test or 
            args.param_test or args.memory_test or args.all):
        print("No tests specified, running comprehensive profile by default")
        args.all = True
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    try:
        model = DCVC_net(lmbda=args.lmbda)
        model.eval()  # Set to evaluation mode
        
        # Create profiler
        profiler = DCVCProfiler(model, device=device)
        
        # Run tests based on arguments
        if args.all:
            profiler.run_comprehensive_profile(output_dir=args.output)
        else:
            # Create standard test frames
            ref_frame, input_frame = profiler.create_random_frames()
            
            if args.layer_test:
                print("\n=== Layer Timing Analysis ===")
                timing_results = profiler.profile_layer_timing(ref_frame, input_frame)
                profiler.display_layer_timing_results(timing_results)
                profiler.visualize_timing_results(timing_results, os.path.join(args.output, "layer_timing.png"))
            
            if args.memory_test and device.type == 'cuda':
                print("\n=== Memory Usage Analysis ===")
                memory_results = profiler.profile_memory_usage(ref_frame, input_frame)
            
            if args.param_test:
                print("\n=== Model Parameter Analysis ===")
                profiler.profile_model_parameters()
            
            if args.res_test:
                print("\n=== Resolution Scaling Analysis ===")
                resolution_results = profiler.profile_resolution_scaling()
                profiler.visualize_resolution_scaling(resolution_results, 
                                                    os.path.join(args.output, "resolution_scaling.png"))
            
            if args.batch_test:
                print("\n=== Batch Size Scaling Analysis ===")
                batch_results = profiler.profile_batch_scaling()
                profiler.visualize_batch_scaling(batch_results, 
                                               os.path.join(args.output, "batch_scaling.png"))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
    #CUDA_VISIBLE_DEVICES=6 python dcvc_profiler.py --all --output profiling_results