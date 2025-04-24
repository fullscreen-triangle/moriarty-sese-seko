from dataclasses import dataclass
import numpy as np
import torch
from typing import Dict, Tuple, List, Any, Union
import multiprocessing as mp
from functools import partial
import logging

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    mass: float
    length: float
    inertia: float


class DynamicsAnalyzer:
    def __init__(self, use_parallel=True, n_processes=None, use_gpu=False):
        self.g = 9.81
        self.segments = {
            'thigh': Segment(mass=7.0, length=0.4, inertia=0.1),
            'shank': Segment(mass=3.5, length=0.4, inertia=0.05),
            'foot': Segment(mass=1.0, length=0.2, inertia=0.01)
        }
        self.use_parallel = use_parallel
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        
        # GPU acceleration setup
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        
        if self.use_gpu:
            logger.info(f"Dynamics analyzer initialized with GPU acceleration on {self.device}")
        else:
            logger.info(f"Dynamics analyzer initialized with parallel={use_parallel}, processes={self.n_processes}")
        
    def calculate_dynamics_batch(self, 
                               positions_batch: List[Dict], 
                               velocities_batch: List[Dict],
                               accelerations_batch: List[Dict]) -> List[Dict]:
        """
        Calculate dynamics for a batch of frames in parallel
        
        Args:
            positions_batch: List of position dictionaries for each frame
            velocities_batch: List of velocity dictionaries for each frame
            accelerations_batch: List of acceleration dictionaries for each frame
            
        Returns:
            List of dynamics results for each frame
        """
        if self.use_gpu:
            return self._calculate_dynamics_batch_gpu(positions_batch, velocities_batch, accelerations_batch)
        
        if not self.use_parallel or len(positions_batch) <= 1:
            # Process sequentially for small batches
            return [self.calculate_dynamics(pos, vel, acc) 
                   for pos, vel, acc in zip(positions_batch, velocities_batch, accelerations_batch)]
        
        # Process in parallel for larger batches
        logger.info(f"Processing {len(positions_batch)} frames in parallel with {self.n_processes} processes")
        
        # Create a partial function with fixed parameters
        process_frame = partial(self._process_single_frame)
        
        # Prepare input data as list of tuples
        input_data = [(pos, vel, acc) for pos, vel, acc in 
                     zip(positions_batch, velocities_batch, accelerations_batch)]
        
        # Process in parallel
        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.map(process_frame, input_data)
            
        return results
    
    def _calculate_dynamics_batch_gpu(self,
                                   positions_batch: List[Dict],
                                   velocities_batch: List[Dict],
                                   accelerations_batch: List[Dict]) -> List[Dict]:
        """
        GPU-accelerated batch processing of dynamics calculations.
        
        This method processes all frames in a single GPU operation for better efficiency.
        """
        logger.info(f"Processing {len(positions_batch)} frames with GPU acceleration")
        
        # Convert batch data to tensors for GPU processing
        batch_size = len(positions_batch)
        segments = ['foot', 'shank', 'thigh']
        
        # Initialize result containers
        forces_batch = [{} for _ in range(batch_size)]
        moments_batch = [{} for _ in range(batch_size)]
        
        # Create initial GRF for all frames in batch
        GRF = torch.tensor([0, 2.5 * 70 * self.g], dtype=torch.float32, device=self.device)
        GRF_batch = GRF.repeat(batch_size, 1)  # Shape: [batch_size, 2]
        
        # Process each segment in sequence (can't parallelize due to dependencies)
        prev_segment = None
        
        for segment in segments:
            seg = self.segments[segment]
            
            # Prepare tensors for batch processing
            if segment == 'foot':
                Fd_batch = GRF_batch
                Md_batch = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            else:
                # Gather proximal forces/moments from previous segment for all frames
                Fd_batch = torch.stack([
                    -torch.tensor(forces_batch[i][prev_segment]['proximal'], dtype=torch.float32, device=self.device)
                    for i in range(batch_size)
                ])
                Md_batch = torch.tensor([
                    -moments_batch[i][prev_segment]['proximal']
                    for i in range(batch_size)
                ], dtype=torch.float32, device=self.device)
            
            # Extract accelerations for current segment across all frames
            com_acc_batch = torch.stack([
                torch.tensor(
                    accelerations_batch[i].get(segment, np.zeros(2)), 
                    dtype=torch.float32, device=self.device
                )
                for i in range(batch_size)
            ])
            
            # Calculate inverse dynamics for entire batch
            segment_mass = torch.tensor(seg.mass, dtype=torch.float32, device=self.device)
            gravity_force = torch.tensor([0, -self.g * seg.mass], dtype=torch.float32, device=self.device)
            gravity_batch = gravity_force.repeat(batch_size, 1)
            
            # Fp = mass * acc - Fd - gravity_force
            Fp_batch = segment_mass * com_acc_batch - Fd_batch - gravity_batch
            
            # Simplified moment calculation
            Mp_batch = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            
            # Store results for this segment
            for i in range(batch_size):
                forces_batch[i][segment] = {
                    'distal': Fd_batch[i].cpu().numpy(),
                    'proximal': Fp_batch[i].cpu().numpy()
                }
                moments_batch[i][segment] = {
                    'distal': Md_batch[i].item(),
                    'proximal': Mp_batch[i].item()
                }
            
            prev_segment = segment
        
        # Combine forces and moments for final output
        results = [
            {'forces': forces_batch[i], 'moments': moments_batch[i]}
            for i in range(batch_size)
        ]
        
        return results
    
    def _process_single_frame(self, frame_data: Tuple[Dict, Dict, Dict]) -> Dict:
        """Process a single frame of data"""
        positions, velocities, accelerations = frame_data
        return self.calculate_dynamics(positions, velocities, accelerations)

    def calculate_dynamics(self, positions: Dict, velocities: Dict,
                           accelerations: Dict) -> Dict:
        if self.use_gpu:
            # Process single frame using GPU by passing it as a batch of 1
            return self._calculate_dynamics_batch_gpu([positions], [velocities], [accelerations])[0]
            
        forces = {}
        moments = {}
        prev_segment = None

        GRF = np.array([0, 2.5 * 70 * self.g])

        for segment in ['foot', 'shank', 'thigh']:
            seg = self.segments[segment]

            if segment == 'foot':
                Fd = GRF
                Md = 0
            else:
                Fd = -forces[prev_segment]['proximal']
                Md = -moments[prev_segment]['proximal']

            Fp, Mp = self._inverse_dynamics(
                positions=positions,
                accelerations=accelerations,
                segment=segment,
                seg_data=seg,
                Fd=Fd,
                Md=Md
            )

            forces[segment] = {'distal': Fd, 'proximal': Fp}
            moments[segment] = {'distal': Md, 'proximal': Mp}
            prev_segment = segment

        return {'forces': forces, 'moments': moments}

    def _inverse_dynamics(self, positions: Dict, accelerations: Dict,
                          segment: str, seg_data: Segment, Fd: np.ndarray,
                          Md: float) -> Tuple[np.ndarray, float]:
        # Inverse dynamics calculation
        com_acc = accelerations.get(segment, np.zeros(2))
        Fp = seg_data.mass * com_acc - Fd - np.array([0, -self.g * seg_data.mass])
        Mp = seg_data.inertia * 0  # Simplified moment calculation

        return Fp, Mp
