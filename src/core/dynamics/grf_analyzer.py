import numpy as np
import torch
from typing import Dict, Tuple, List, Union


class GRFAnalyzer:
    def __init__(self, fps=30.0, use_gpu=False):
        """
        Initialize the GRF analyzer.
        
        Args:
            fps: Frames per second for temporal calculations
            use_gpu: Whether to use GPU acceleration
        """
        self.g = 9.81
        self.body_mass = 70  # Default mass in kg
        self.fps = fps
        
        # GPU setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        
        if self.use_gpu:
            print(f"GRF Analyzer using GPU acceleration on {self.device}")

    def estimate_grf(self, positions: Dict, accelerations: Dict) -> Dict:
        """
        Estimate ground reaction forces from kinematic data.
        
        Args:
            positions: Joint positions dictionary
            accelerations: Accelerations dictionary
            
        Returns:
            Dictionary containing GRF estimates
        """
        if self.use_gpu:
            return self._estimate_grf_gpu(positions, accelerations)
            
        # CPU implementation (original)
        # Vertical GRF estimation
        vertical_grf = self._estimate_vertical_grf(accelerations)

        # Horizontal GRF estimation
        horizontal_grf = self._estimate_horizontal_grf(accelerations)

        # Impact forces
        impact_force = self._estimate_impact_force(positions)

        return {
            'vertical_grf': vertical_grf,
            'horizontal_grf': horizontal_grf,
            'impact_force': impact_force
        }
        
    def estimate_grf_batch(self, positions_batch: List[Dict], accelerations_batch: List[Dict]) -> List[Dict]:
        """
        Estimate GRF for a batch of frames.
        
        Args:
            positions_batch: List of position dictionaries
            accelerations_batch: List of acceleration dictionaries
            
        Returns:
            List of GRF dictionaries for each frame
        """
        if self.use_gpu:
            return self._estimate_grf_batch_gpu(positions_batch, accelerations_batch)
            
        # CPU batch processing
        results = []
        for positions, accelerations in zip(positions_batch, accelerations_batch):
            grf = self.estimate_grf(positions, accelerations)
            results.append(grf)
            
        return results
        
    def _estimate_grf_gpu(self, positions: Dict, accelerations: Dict) -> Dict:
        """
        GPU-accelerated GRF estimation for a single frame.
        
        Args:
            positions: Joint positions dictionary
            accelerations: Accelerations dictionary
            
        Returns:
            Dictionary containing GRF estimates
        """
        # Convert to tensors and move to GPU
        com_acc = torch.tensor(
            accelerations.get('com', np.zeros(2)), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Calculate vertical GRF: mass * (g + vertical_acceleration)
        vertical_grf = self.body_mass * (self.g + com_acc[1])
        
        # Calculate horizontal GRF: mass * horizontal_acceleration
        horizontal_grf = self.body_mass * com_acc[0]
        
        # Calculate impact force
        impact_force = 2.5 * self.body_mass * self.g
        
        # Return results as a dictionary with numpy values
        return {
            'vertical_grf': vertical_grf.item(),
            'horizontal_grf': horizontal_grf.item(),
            'impact_force': impact_force
        }
        
    def _estimate_grf_batch_gpu(self, positions_batch: List[Dict], accelerations_batch: List[Dict]) -> List[Dict]:
        """
        GPU-accelerated GRF estimation for a batch of frames.
        
        Args:
            positions_batch: List of position dictionaries
            accelerations_batch: List of acceleration dictionaries
            
        Returns:
            List of GRF dictionaries for each frame
        """
        batch_size = len(positions_batch)
        
        # Extract COM accelerations across all frames
        com_accs = []
        for acc in accelerations_batch:
            com_accs.append(acc.get('com', np.zeros(2)))
            
        # Convert to tensor and move to GPU
        com_accs_tensor = torch.tensor(com_accs, dtype=torch.float32, device=self.device)
        
        # Calculate vertical GRF for all frames: mass * (g + vertical_acceleration)
        vertical_grfs = self.body_mass * (self.g + com_accs_tensor[:, 1])
        
        # Calculate horizontal GRF for all frames: mass * horizontal_acceleration
        horizontal_grfs = self.body_mass * com_accs_tensor[:, 0]
        
        # Calculate impact force (constant)
        impact_force = 2.5 * self.body_mass * self.g
        
        # Prepare results for each frame
        results = []
        for i in range(batch_size):
            results.append({
                'vertical_grf': vertical_grfs[i].item(),
                'horizontal_grf': horizontal_grfs[i].item(),
                'impact_force': impact_force
            })
            
        return results

    def _estimate_vertical_grf(self, accelerations: Dict) -> float:
        # Simple estimation based on acceleration
        com_acc = accelerations.get('com', np.zeros(2))
        return self.body_mass * (self.g + com_acc[1])

    def _estimate_horizontal_grf(self, accelerations: Dict) -> float:
        com_acc = accelerations.get('com', np.zeros(2))
        return self.body_mass * com_acc[0]

    def _estimate_impact_force(self, positions: Dict) -> float:
        # Simplified impact force estimation
        return 2.5 * self.body_mass * self.g
        
    def analyze(self, pose_sequence: List[Dict]) -> Dict:
        """
        Analyze ground reaction forces for a sequence of poses.
        
        Args:
            pose_sequence: List of pose dictionaries
            
        Returns:
            Dictionary with GRF analysis results
        """
        # Extract positions and calculate accelerations
        positions = [pose.get('positions', {}) for pose in pose_sequence]
        
        # Simple acceleration calculation using finite differences
        accelerations = []
        for i in range(len(positions)):
            if i < 2:
                # For first frames, use zero acceleration
                accelerations.append({'com': np.zeros(2)})
            else:
                # Calculate acceleration using previous positions
                # This is a simplified calculation
                p_current = positions[i].get('com', np.zeros(2))
                p_prev = positions[i-1].get('com', np.zeros(2))
                p_prev2 = positions[i-2].get('com', np.zeros(2))
                
                # Second derivative approximation
                acc = (p_current - 2 * p_prev + p_prev2) * (self.fps ** 2)
                accelerations.append({'com': acc})
        
        # Estimate GRF for each frame
        if self.use_gpu:
            grf_data = self._estimate_grf_batch_gpu(positions, accelerations)
        else:
            grf_data = []
            for p, a in zip(positions, accelerations):
                grf_data.append(self.estimate_grf(p, a))
        
        # Calculate average, peak values
        vertical_grfs = [frame['vertical_grf'] for frame in grf_data]
        horizontal_grfs = [frame['horizontal_grf'] for frame in grf_data]
        
        # Calculate statistics
        grf_stats = {
            'vertical_grf': {
                'mean': np.mean(vertical_grfs),
                'max': np.max(vertical_grfs),
                'min': np.min(vertical_grfs)
            },
            'horizontal_grf': {
                'mean': np.mean(horizontal_grfs),
                'max': np.max(horizontal_grfs),
                'min': np.min(horizontal_grfs)
            },
            'frames': grf_data
        }
        
        return grf_stats
        
    def plot_grf(self, grf_data: Dict, output_path: str) -> None:
        """
        Plot ground reaction force data.
        
        Args:
            grf_data: GRF data to plot
            output_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            if 'frames' in grf_data:
                vertical_grfs = [frame['vertical_grf'] for frame in grf_data['frames']]
                horizontal_grfs = [frame['horizontal_grf'] for frame in grf_data['frames']]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot data
                ax.plot(vertical_grfs, label='Vertical GRF')
                ax.plot(horizontal_grfs, label='Horizontal GRF')
                
                # Add labels and title
                ax.set_xlabel('Frame')
                ax.set_ylabel('Force (N)')
                ax.set_title('Ground Reaction Forces')
                ax.legend()
                
                # Save figure
                plt.savefig(output_path)
                plt.close()
            else:
                # Handle multiple athletes
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for athlete_id, data in grf_data.items():
                    if 'vertical_grf' in data and 'mean' in data['vertical_grf']:
                        ax.bar(f"{athlete_id}_vert", data['vertical_grf']['mean'], 
                              yerr=data['vertical_grf']['max'] - data['vertical_grf']['mean'],
                              label=f"Athlete {athlete_id} (Vertical)")
                        ax.bar(f"{athlete_id}_horz", data['horizontal_grf']['mean'],
                              yerr=data['horizontal_grf']['max'] - data['horizontal_grf']['mean'],
                              label=f"Athlete {athlete_id} (Horizontal)")
                
                ax.set_xlabel('Athlete')
                ax.set_ylabel('Force (N)')
                ax.set_title('Average Ground Reaction Forces by Athlete')
                ax.legend()
                
                plt.savefig(output_path)
                plt.close()
                
        except ImportError:
            print("Matplotlib is required for plotting GRF data")
        except Exception as e:
            print(f"Error plotting GRF data: {e}")
