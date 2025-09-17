#!/usr/bin/env python3
"""
Test script to verify wandb setup and configuration.
"""

import sys
from pathlib import Path

# Add audiocraft to path
sys.path.insert(0, str(Path(__file__).parent))

import wandb
import omegaconf
import hydra
from hydra import initialize, compose


def test_wandb_setup():
    """Test wandb setup and configuration."""
    
    print("Testing wandb setup...")
    
    # Test wandb import
    try:
        print(f"✓ Wandb version: {wandb.__version__}")
    except Exception as e:
        print(f"✗ Failed to import wandb: {e}")
        return False
    
    # Test wandb login status
    try:
        api = wandb.Api()
        print("✓ Wandb API accessible")
    except Exception as e:
        print(f"✗ Wandb API not accessible: {e}")
        print("Please run 'wandb login' to authenticate")
        return False
    
    # Test configuration loading
    try:
        with initialize(config_path="config", version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "solver=compression/encodec_eigenscape_1ch",
                    "dset=eigenscape_1ch",
                    "dataset_type=eigenscape",
                    "logging.log_wandb=true",
                    "wandb.project=test-project",
                    "wandb.name=test-run",
                ]
            )
            print("✓ Configuration loaded successfully")
            print(f"  - Wandb logging enabled: {cfg.logging.log_wandb}")
            print(f"  - Wandb project: {cfg.wandb.project}")
            print(f"  - Wandb name: {cfg.wandb.name}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # Test wandb initialization
    try:
        run = wandb.init(
            project="test-project",
            name="test-run",
            config={"test": "value"},
            mode="disabled"  # Use disabled mode for testing
        )
        print("✓ Wandb run initialized successfully")
        wandb.finish()
    except Exception as e:
        print(f"✗ Failed to initialize wandb run: {e}")
        return False
    
    print("\n✓ All wandb tests passed!")
    return True


if __name__ == "__main__":
    success = test_wandb_setup()
    sys.exit(0 if success else 1)
