# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import random

from .audio_dataset import AudioDataset


class EigenscapeDataset(AudioDataset):
    """Dataset for eigenscape multichannel audio data with 1-channel mode support.
    
    This dataset integrates the baseline dataloader functionality with AudioCraft's
    dataset interface for training EnCodec models.
    """
    
    def __init__(self, 
                 audio_dir: str,
                 sample_rate: int = 24000,
                 segment_duration: float = 1.0,
                 channels: int = 1,
                 dataset_size: int = None,
                 pad: bool = True,
                 random_crop: bool = True,
                 file_extensions: tp.List[str] = None,
                 min_file_duration: float = None,
                 train_folders: tp.List[str] = None,
                 val_folders: tp.List[str] = None,
                 cache_size: int = 100,
                 **kwargs):
        """
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            segment_duration: Duration of segments in seconds
            channels: Number of channels to load (1 for 1-channel mode)
            dataset_size: Virtual dataset size (if None = number of files)
            pad: Whether to pad shorter segments
            random_crop: Whether to randomly crop longer segments
            file_extensions: List of file extensions to include
            min_file_duration: Minimum file duration to include (in seconds)
            train_folders: List of folder names to use for training
            val_folders: List of folder names to use for validation
            cache_size: Number of files to cache in memory
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.channels = channels
        self.pad = pad
        self.cache_size = cache_size
        self._audio_cache = {}
        self.random_crop = random_crop
        self.file_extensions = file_extensions or ['.wav', '.WAV']
        self.min_file_duration = min_file_duration
        self.train_folders = train_folders or []
        self.val_folders = val_folders or []
        
        # Find audio files
        self.audio_files = self._find_audio_files()
        
        # Filter by duration if specified
        if min_file_duration is not None:
            self.audio_files = self._filter_by_duration(self.audio_files, min_file_duration)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        # Set dataset size
        self.dataset_size = dataset_size or len(self.audio_files)
        
        # Calculate segment length in samples
        self.segment_length = int(segment_duration * sample_rate)
    
    def _find_audio_files(self) -> tp.List[Path]:
        """Find all audio files in the specified directories."""
        files = []
        
        # If specific folders are specified, only look in those
        if self.train_folders or self.val_folders:
            folders_to_search = self.train_folders + self.val_folders
            for folder in folders_to_search:
                folder_path = self.audio_dir / folder
                if folder_path.exists():
                    for ext in self.file_extensions:
                        files.extend(folder_path.glob(f"**/*{ext}"))
                        files.extend(folder_path.glob(f"**/*{ext.upper()}"))
        else:
            # Search all subdirectories
            for ext in self.file_extensions:
                files.extend(self.audio_dir.glob(f"**/*{ext}"))
                files.extend(self.audio_dir.glob(f"**/*{ext.upper()}"))
        
        return sorted(files)
    
    def _filter_by_duration(self, files: tp.List[Path], min_duration: float) -> tp.List[Path]:
        """Filter files by minimum duration."""
        valid_files = []
        for file_path in files:
            try:
                info = sf.info(str(file_path))
                if info.duration >= min_duration:
                    valid_files.append(file_path)
            except Exception:
                continue
        return valid_files
    
    def _get_file_duration(self, file_path: Path) -> float:
        """Get duration of a file in seconds."""
        try:
            info = sf.info(str(file_path))
            return info.duration
        except Exception:
            return 0.0
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def _load_audio_file(self, file_path: Path) -> torch.Tensor:
        """Load and cache audio file."""
        file_str = str(file_path)
        
        # Check cache first
        if file_str in self._audio_cache:
            return self._audio_cache[file_str]
        
        # Load audio file
        try:
            audio, sr = sf.read(file_str, dtype=np.float32)
            
            # Handle different audio shapes
            if len(audio.shape) == 1:
                # Mono audio - tile to match channel count
                audio = np.tile(audio, (self.channels, 1))
            else:
                # Multi-channel audio - transpose to (channels, samples)
                audio = audio.T
            
            # Fix channel count
            if audio.shape[0] < self.channels:
                # Pad with zeros if not enough channels
                padding = np.zeros((self.channels - audio.shape[0], audio.shape[1]), dtype=np.float32)
                audio = np.vstack([audio, padding])
            elif audio.shape[0] > self.channels:
                if self.channels == 1:
                    # For 1-channel mode, randomly pick one of the available channels
                    random_channel_idx = random.randint(0, audio.shape[0] - 1)
                    audio = audio[random_channel_idx:random_channel_idx + 1, :]
                else:
                    # For multi-channel mode, take the first N channels
                    audio = audio[:self.channels, :]
            
            audio = torch.from_numpy(audio).float()
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
            
            # Cache the audio (with LRU eviction if cache is full)
            if len(self._audio_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
            
            self._audio_cache[file_str] = audio
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if loading fails
            return torch.zeros((self.channels, self.segment_length), dtype=torch.float32)
    
    def _resample(self, audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio using torchaudio."""
        try:
            import torchaudio
            # Resample each channel
            resampled_channels = []
            for ch in range(audio.shape[0]):
                resampled = torchaudio.functional.resample(
                    audio[ch:ch+1], orig_sr, target_sr
                )
                resampled_channels.append(resampled)
            return torch.cat(resampled_channels, dim=0)
        except ImportError:
            # Fallback to simple resampling if torchaudio not available
            ratio = target_sr / orig_sr
            new_length = int(audio.shape[1] * ratio)
            resampled = torch.nn.functional.interpolate(
                audio.unsqueeze(0), size=new_length, mode='linear', align_corners=False
            )
            return resampled.squeeze(0)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a random segment from a random file."""
        # Pick a random file
        file_path = random.choice(self.audio_files)
        audio = self._load_audio_file(file_path)
        
        # Get segment
        if audio.shape[1] < self.segment_length:
            # Pad if too short
            if self.pad:
                padding = torch.zeros((self.channels, self.segment_length - audio.shape[1]))
                audio = torch.cat([audio, padding], dim=1)
            else:
                # Repeat audio to fill segment
                repeats = (self.segment_length // audio.shape[1]) + 1
                audio = audio.repeat(1, repeats)
                audio = audio[:, :self.segment_length]
        elif audio.shape[1] > self.segment_length:
            # Crop if too long
            if self.random_crop:
                start = random.randint(0, audio.shape[1] - self.segment_length)
            else:
                start = 0
            audio = audio[:, start:start + self.segment_length]
        
        # Ensure we have the right shape: (channels, samples)
        assert audio.shape == (self.channels, self.segment_length), \
            f"Expected shape ({self.channels}, {self.segment_length}), got {audio.shape}"
        
        return audio


class FixedEigenscapeDataset(EigenscapeDataset):
    """Fixed validation dataset that uses the same segments every epoch."""
    
    def __init__(self, 
                 audio_files: tp.List[Path],
                 sample_rate: int = 24000,
                 segment_duration: float = 1.0,
                 channels: int = 1,
                 min_file_duration: float = None,
                 random_crop: bool = True,
                 **kwargs):
        """
        Args:
            audio_files: Pre-selected list of audio files to use
            sample_rate: Target sample rate
            segment_duration: Duration of segments in seconds
            channels: Number of channels to load (1 for 1-channel mode)
            min_file_duration: Minimum file duration to include (in seconds)
            random_crop: Whether to randomly crop longer segments (only for initial setup)
        """
        # Set up basic attributes first
        self.audio_dir = Path("")  # Not used since we provide files directly
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.channels = channels
        self.pad = kwargs.get('pad', True)
        self.cache_size = kwargs.get('cache_size', 100)
        self._audio_cache = {}
        self.random_crop = random_crop
        self.file_extensions = kwargs.get('file_extensions', ['.wav', '.WAV'])
        self.min_file_duration = min_file_duration
        self.train_folders = kwargs.get('train_folders', [])
        self.val_folders = kwargs.get('val_folders', [])
        
        # Set provided files directly
        self.audio_files = audio_files
        
        # Set dataset size
        self.dataset_size = len(audio_files)
        
        # Calculate segment length in samples
        self.segment_length = int(segment_duration * sample_rate)
        
        # Pre-select fixed segments and channels for each file
        self._fixed_segments = []
        self._fixed_channels = []
        
        for file_path in audio_files:
            # Load the file once to get its properties
            try:
                audio, sr = sf.read(str(file_path), dtype=np.float32)
                if len(audio.shape) == 1:
                    audio = np.tile(audio, (32, 1))  # Assume max 32 channels
                else:
                    audio = audio.T
                
                # Resample if needed
                if sr != sample_rate:
                    audio_tensor = torch.from_numpy(audio).float()
                    audio_tensor = self._resample(audio_tensor, sr, sample_rate)
                    audio = audio_tensor.numpy()
                
                # Pre-select a random segment and channel for this file
                if audio.shape[1] >= self.segment_length:
                    if random_crop:
                        start = random.randint(0, audio.shape[1] - self.segment_length)
                    else:
                        start = 0
                    segment_start = start
                else:
                    segment_start = 0
                
                # Pre-select a random channel (for 1-channel mode)
                if channels == 1 and audio.shape[0] > 1:
                    channel_idx = random.randint(0, audio.shape[0] - 1)
                else:
                    channel_idx = 0
                
                self._fixed_segments.append(segment_start)
                self._fixed_channels.append(channel_idx)
                
            except Exception as e:
                print(f"Error pre-processing {file_path}: {e}")
                # Use defaults
                self._fixed_segments.append(0)
                self._fixed_channels.append(0)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the same fixed segment from the same file every time."""
        # Use the pre-selected file, segment, and channel
        file_path = self.audio_files[idx]
        segment_start = self._fixed_segments[idx]
        channel_idx = self._fixed_channels[idx]
        
        # Load the file
        audio = self._load_audio_file(file_path)
        
        # Use the pre-selected channel
        if self.channels == 1 and audio.shape[0] > 1:
            audio = audio[channel_idx:channel_idx + 1, :]
        
        # Use the pre-selected segment
        if audio.shape[1] < self.segment_length:
            # Pad if too short
            if self.pad:
                padding = torch.zeros((self.channels, self.segment_length - audio.shape[1]))
                audio = torch.cat([audio, padding], dim=1)
            else:
                # Repeat audio to fill segment
                repeats = (self.segment_length // audio.shape[1]) + 1
                audio = audio.repeat(1, repeats)
                audio = audio[:, :self.segment_length]
        elif audio.shape[1] > self.segment_length:
            # Use the pre-selected segment
            audio = audio[:, segment_start:segment_start + self.segment_length]
        
        # Ensure we have the right shape: (channels, samples)
        assert audio.shape == (self.channels, self.segment_length), \
            f"Expected shape ({self.channels}, {self.segment_length}), got {audio.shape}"
        
        return audio


def create_eigenscape_dataloader(audio_dir: str,
                                batch_size: int = 32,
                                sample_rate: int = 24000,
                                segment_duration: float = 1.0,
                                channels: int = 1,
                                num_workers: int = 8,
                                shuffle: bool = True,
                                dataset_size: int = None,
                                train_folders: tp.List[str] = None,
                                val_folders: tp.List[str] = None,
                                min_file_duration: float = None,
                                **kwargs) -> DataLoader:
    """Create a DataLoader for eigenscape audio data.
    
    Args:
        audio_dir: Directory containing audio files
        batch_size: Batch size for training
        sample_rate: Target sample rate
        segment_duration: Duration of segments in seconds
        channels: Number of channels to load (1 for 1-channel mode)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        dataset_size: Virtual dataset size (if None = number of files)
        train_folders: List of folder names to use for training
        val_folders: List of folder names to use for validation
        min_file_duration: Minimum file duration to include (in seconds)
        **kwargs: Additional arguments passed to EigenscapeDataset
    """
    
    dataset = EigenscapeDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=dataset_size,
        train_folders=train_folders,
        val_folders=val_folders,
        min_file_duration=min_file_duration,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    return dataloader


def create_eigenscape_train_val_dataloaders(audio_dir: str,
                                           batch_size: int = 32,
                                           sample_rate: int = 24000,
                                           segment_duration: float = 1.0,
                                           channels: int = 1,
                                           num_workers: int = 8,
                                           train_dataset_size: int = 8000,
                                           val_dataset_size: int = 2000,
                                           train_folders: tp.List[str] = None,
                                           val_folders: tp.List[str] = None,
                                           min_file_duration: float = None,
                                           **kwargs) -> tp.Tuple[DataLoader, DataLoader]:
    """Create both training and validation dataloaders for eigenscape data.
    
    Training uses random sampling (fresh segments, files, channels each epoch).
    Validation uses fixed sampling (same segments, files, channels each epoch).
    
    Args:
        audio_dir: Directory containing audio files
        batch_size: Batch size for training
        sample_rate: Target sample rate
        segment_duration: Duration of segments in seconds
        channels: Number of channels to load (1 for 1-channel mode)
        num_workers: Number of worker processes
        train_dataset_size: Virtual training dataset size
        val_dataset_size: Fixed validation dataset size
        train_folders: List of folder names to use for training
        val_folders: List of folder names to use for validation
        min_file_duration: Minimum file duration to include (in seconds)
        **kwargs: Additional arguments passed to datasets
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from pathlib import Path
    
    audio_dir_path = Path(audio_dir)
    file_extensions = kwargs.get('file_extensions', ['.wav', '.WAV'])
    file_extensions = [ext.lower() for ext in file_extensions]
    
    # Find training files from specified folders
    train_files = []
    if train_folders:
        for folder in train_folders:
            folder_path = audio_dir_path / folder
            if folder_path.exists():
                for ext in file_extensions:
                    train_files.extend(folder_path.glob(f"**/*{ext}"))
                    train_files.extend(folder_path.glob(f"**/*{ext.upper()}"))
    
    # Find validation files from specified folders
    val_files = []
    if val_folders:
        for folder in val_folders:
            folder_path = audio_dir_path / folder
            if folder_path.exists():
                for ext in file_extensions:
                    val_files.extend(folder_path.glob(f"**/*{ext}"))
                    val_files.extend(folder_path.glob(f"**/*{ext.upper()}"))
    
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    
    if len(train_files) == 0:
        raise ValueError(f"No training files found in folders: {train_folders}")
    if len(val_files) == 0:
        raise ValueError(f"No validation files found in folders: {val_folders}")
    
    # Filter files by duration if specified
    if min_file_duration is not None:
        train_files = _filter_files_by_duration(train_files, min_file_duration, sample_rate)
        val_files = _filter_files_by_duration(val_files, min_file_duration, sample_rate)
        
        if len(train_files) == 0:
            raise ValueError(f"No training files longer than {min_file_duration}s found")
        if len(val_files) == 0:
            raise ValueError(f"No validation files longer than {min_file_duration}s found")
    
    print(f"Found {len(train_files)} training files from folders: {train_folders}")
    print(f"Found {len(val_files)} validation files from folders: {val_folders}")
    
    # Create training dataset (random sampling)
    train_dataset = EigenscapeDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        dataset_size=train_dataset_size,
        train_folders=train_folders,
        val_folders=val_folders,
        min_file_duration=min_file_duration,
        **kwargs
    )
    # Override the audio_files with our training files
    train_dataset.audio_files = train_files
    
    # Create fixed validation dataset (same segments every epoch)
    val_dataset = FixedEigenscapeDataset(
        audio_files=val_files,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        channels=channels,
        min_file_duration=min_file_duration,
        random_crop=kwargs.get('random_crop', True)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=kwargs.get('pin_memory', True),
        persistent_workers=kwargs.get('persistent_workers', num_workers > 0),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data for consistent evaluation
        num_workers=num_workers,
        pin_memory=kwargs.get('pin_memory', True),
        persistent_workers=kwargs.get('persistent_workers', num_workers > 0),
        drop_last=True
    )
    
    return train_loader, val_loader


def _filter_files_by_duration(files: tp.List[Path], min_duration: float, sample_rate: int) -> tp.List[Path]:
    """Filter files by minimum duration."""
    valid_files = []
    for file_path in files:
        try:
            info = sf.info(str(file_path))
            if info.duration >= min_duration:
                valid_files.append(file_path)
        except Exception:
            continue
    return valid_files
