# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid search file for training EnCodec on eigenscape data in 1-channel mode.

This grid shows how to train an EnCodec model on the eigenscape dataset
using the baseline dataloader with 1-channel audio.
"""

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=8, partition=partitions)
    
    # EnCodec trained on eigenscape data in 1-channel mode
    launcher.bind_(solver='compression/encodec_eigenscape_1ch')
    launcher.bind_(dset='eigenscape_1ch')
    
    # Set dataset type to eigenscape
    launcher.bind_(dataset_type='eigenscape')
    
    # Set folder configuration for eigenscape data
    launcher.bind_(
        train_folders=["Airport", "Bus", "Metro", "Metro Station", "Park", "Public Square", "Shopping Centre"],
        val_folders=["Woodland", "Train Station"],
        min_file_duration=1.0
    )
    
    # launch xp
    launcher()
    
    # Optional: Add evaluation with ViSQOL if available
    launcher({
        'metrics.visqol.bin': '/data/home/jadecopet/local/usr/opt/visqol',
        'label': 'visqol',
        'evaluate.metrics.visqol': True
    })
