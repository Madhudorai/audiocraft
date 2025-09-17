#!/usr/bin/env python3
"""
Test script for eigenscape dataloader.

This script tests the eigenscape dataloader to ensure it works correctly
with the /scratch/eigenscape data.
"""

import sys
from pathlib import Path

# Add audiocraft to path
sys.path.insert(0, str(Path(__file__).parent))

from audiocraft.data.eigenscape_dataset import create_eigenscape_dataloader
import torch


def test_eigenscape_dataloader():
    """Test the eigenscape dataloader."""
    
    print("Testing eigenscape dataloader...")
    
    # Create dataloader
    dataloader = create_eigenscape_dataloader(
        audio_dir="/scratch/eigenscape",
        batch_size=4,
        sample_rate=24000,
        segment_duration=1.0,
        channels=1,  # 1-channel mode
        num_workers=2,
        shuffle=True,
        dataset_size=100,  # Small dataset for testing
        train_folders=["Airport", "Bus", "Metro", "Metro Station", "Park", "Public Square", "Shopping Centre"],
        val_folders=["Woodland", "Train Station"],
        min_file_duration=1.0
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Test loading a few batches
    print("\nTesting batch loading...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}")
        print(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}")
        print(f"  Mean: {batch.mean():.4f}, Std: {batch.std():.4f}")
        
        # Check shape
        expected_shape = (4, 1, 24000)  # (batch_size, channels, samples)
        if batch.shape != expected_shape:
            print(f"  WARNING: Expected shape {expected_shape}, got {batch.shape}")
        else:
            print(f"  ✓ Shape correct: {batch.shape}")
        
        if i >= 2:  # Test only first 3 batches
            break
    
    print("\n✓ Eigenscape dataloader test completed successfully!")


if __name__ == "__main__":
    test_eigenscape_dataloader()
