#!/usr/bin/env python3
"""
Training script for EnCodec on eigenscape data in 1-channel mode.

This script demonstrates how to train an EnCodec model using the baseline
dataloader with eigenscape data from /scratch/eigenscape.
"""

import sys
import os
from pathlib import Path

# Add audiocraft to path
sys.path.insert(0, str(Path(__file__).parent))

from audiocraft.train import get_solver, init_seed_and_system
from audiocraft.solvers.builders import get_solver as get_solver_builder
import omegaconf
import hydra
from hydra import initialize, compose
import flashy
import wandb


def train_encodec_eigenscape():
    """Train EnCodec on eigenscape data."""
    
    # Set up wandb environment variables
    os.environ.setdefault('WANDB_MODE', 'online')
    os.environ.setdefault('WANDB_SILENT', 'false')
    
    # Initialize Hydra
    with initialize(config_path="config", version_base="1.1"):
        # Compose configuration
        cfg = compose(
            config_name="config",
            overrides=[
                "solver=compression/encodec_eigenscape_1ch",
                "dset=eigenscape_1ch",
                "dataset_type=eigenscape",
                "train_folders=[Airport,Bus,Metro,Metro Station,Park,Public Square,Shopping Centre]",
                "val_folders=[Woodland,Train Station]",
                "min_file_duration=1.0",
                "optim.epochs=10",  # Reduced for testing
                "optim.updates_per_epoch=100",  # Reduced for testing
                "dataset.train.num_samples=1000",  # Reduced for testing
                "dataset.valid.num_samples=200",  # Reduced for testing
                # Enable wandb logging
                "logging.log_wandb=true",
                "wandb.project=audiocraft-encodec-eigenscape",
                "wandb.name=encodec-eigenscape-1ch",
                "wandb.group=eigenscape-experiments",
                "wandb.with_media_logging=true",
            ]
        )
        
        print("Configuration loaded successfully!")
        print(f"Sample rate: {cfg.sample_rate}")
        print(f"Channels: {cfg.channels}")
        print(f"Dataset type: {cfg.dataset_type}")
        print(f"Training folders: {cfg.train_folders}")
        print(f"Validation folders: {cfg.val_folders}")
        print(f"Wandb logging enabled: {cfg.logging.log_wandb}")
        if cfg.logging.log_wandb:
            print(f"Wandb project: {cfg.wandb.project}")
            print(f"Wandb name: {cfg.wandb.name}")
            print(f"Wandb group: {cfg.wandb.group}")
        
        # Initialize system and seed
        init_seed_and_system(cfg)
        
        # Setup logging
        log_name = '%s.log.{rank}' % cfg.execute_only if cfg.execute_only else 'solver.log.{rank}'
        flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=log_name)
        
        # Initialize distributed training
        flashy.distrib.init()
        
        # Get solver and run training
        solver = get_solver(cfg)
        if cfg.show:
            solver.show()
            return

        if cfg.execute_only:
            assert cfg.execute_inplace or cfg.continue_from is not None, \
                "Please explicitly specify the checkpoint to continue from with continue_from=<sig_or_path> " + \
                "when running with execute_only or set execute_inplace to True."
            solver.restore(replay_metrics=False)  # load checkpoint
            solver.run_one_stage(cfg.execute_only)
            return

        # Run training with wandb logging
        try:
            result = solver.run()
            print("Training completed successfully!")
            return result
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        finally:
            # Ensure wandb is properly closed
            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    train_encodec_eigenscape()
