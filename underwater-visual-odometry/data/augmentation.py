"""
Underwater-specific data augmentation for visual odometry

Applies augmentations that preserve spatial relationships between frames
while enhancing model robustness to underwater conditions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List
import random


class UnderwaterAugmentation:
    """
    Underwater-specific augmentation pipeline
    
    Applies consistent augmentations across all cameras and frames in a sequence
    to preserve spatial and temporal relationships
    """
    
    def __init__(
        self,
        color_jitter_prob: float = 0.8,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        blur_prob: float = 0.2,
        blur_kernel_range: Tuple[int, int] = (3, 7),
        underwater_distortion_prob: float = 0.4,
        lighting_change_prob: float = 0.5
    ):
        """
        Args:
            color_jitter_prob: Probability of applying color jittering
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment  
            saturation_range: Range for saturation adjustment
            hue_range: Range for hue adjustment
            noise_prob: Probability of adding noise
            noise_std: Standard deviation of Gaussian noise
            blur_prob: Probability of applying blur
            blur_kernel_range: Range for blur kernel sizes
            underwater_distortion_prob: Probability of underwater-specific distortions
            lighting_change_prob: Probability of lighting changes
        """
        self.color_jitter_prob = color_jitter_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.underwater_distortion_prob = underwater_distortion_prob
        self.lighting_change_prob = lighting_change_prob
        
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a sequence of multi-camera images
        
        Args:
            images: Input images [seq_len, num_cameras, 3, H, W]
        Returns:
            Augmented images [seq_len, num_cameras, 3, H, W]
        """
        seq_len, num_cameras, channels, height, width = images.shape
        
        # Sample augmentation parameters (consistent across sequence and cameras)
        aug_params = self._sample_augmentation_parameters()
        
        # Apply augmentations
        augmented_images = []
        
        for t in range(seq_len):
            frame_cameras = []
            
            for c in range(num_cameras):
                img = images[t, c]  # [3, H, W]
                
                # Apply augmentations with consistent parameters
                aug_img = self._apply_augmentations(img, aug_params)
                frame_cameras.append(aug_img)
            
            augmented_images.append(torch.stack(frame_cameras))
        
        return torch.stack(augmented_images)
    
    def _sample_augmentation_parameters(self) -> dict:
        """Sample augmentation parameters for consistent application"""
        params = {}
        
        # Color jittering parameters
        if random.random() < self.color_jitter_prob:
            params['brightness_factor'] = random.uniform(*self.brightness_range)
            params['contrast_factor'] = random.uniform(*self.contrast_range)
            params['saturation_factor'] = random.uniform(*self.saturation_range)
            params['hue_factor'] = random.uniform(*self.hue_range)
            params['apply_color_jitter'] = True
        else:
            params['apply_color_jitter'] = False
        
        # Noise parameters
        params['apply_noise'] = random.random() < self.noise_prob
        
        # Blur parameters
        if random.random() < self.blur_prob:
            kernel_size = random.randrange(*self.blur_kernel_range, 2)  # Odd kernel sizes
            if kernel_size % 2 == 0:
                kernel_size += 1
            params['blur_kernel_size'] = kernel_size
            params['apply_blur'] = True
        else:
            params['apply_blur'] = False
        
        # Underwater distortion parameters
        params['apply_underwater_distortion'] = random.random() < self.underwater_distortion_prob
        if params['apply_underwater_distortion']:
            params['distortion_strength'] = random.uniform(0.1, 0.3)
        
        # Lighting change parameters
        params['apply_lighting_change'] = random.random() < self.lighting_change_prob
        if params['apply_lighting_change']:
            params['lighting_direction'] = random.choice(['top', 'bottom', 'left', 'right'])
            params['lighting_strength'] = random.uniform(0.1, 0.4)
        
        return params
    
    def _apply_augmentations(
        self, 
        image: torch.Tensor, 
        params: dict
    ) -> torch.Tensor:
        """
        Apply augmentations to a single image
        
        Args:
            image: Input image [3, H, W]
            params: Augmentation parameters
        Returns:
            Augmented image [3, H, W]
        """
        img = image.clone()
        
        # 1. Color jittering
        if params['apply_color_jitter']:
            img = self._color_jitter(
                img,
                brightness=params['brightness_factor'],
                contrast=params['contrast_factor'],
                saturation=params['saturation_factor'],
                hue=params['hue_factor']
            )
        
        # 2. Gaussian noise
        if params['apply_noise']:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0, 1)
        
        # 3. Blur
        if params['apply_blur']:
            img = self._gaussian_blur(img, params['blur_kernel_size'])
        
        # 4. Underwater-specific distortions
        if params['apply_underwater_distortion']:
            img = self._underwater_distortion(img, params['distortion_strength'])
        
        # 5. Lighting changes
        if params['apply_lighting_change']:
            img = self._lighting_change(
                img, 
                params['lighting_direction'], 
                params['lighting_strength']
            )
        
        return img
    
    def _color_jitter(
        self,
        image: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float
    ) -> torch.Tensor:
        """Apply color jittering to image"""
        img = image.clone()
        
        # Convert to HSV for better color manipulation
        # Note: This is a simplified version - in practice you might want to use
        # proper RGB to HSV conversion
        
        # Brightness adjustment
        img = img * brightness
        
        # Contrast adjustment (around mean)
        mean = img.mean(dim=[1, 2], keepdim=True)
        img = (img - mean) * contrast + mean
        
        # Clamp to valid range
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def _gaussian_blur(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply Gaussian blur to image"""
        # Create Gaussian kernel
        sigma = kernel_size / 6.0  # Rule of thumb
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(image.device)
        
        # Apply convolution with padding
        padding = kernel_size // 2
        
        # Apply to each channel separately
        channels = []
        for c in range(image.shape[0]):
            channel = image[c:c+1].unsqueeze(0)  # [1, 1, H, W]
            blurred = F.conv2d(channel, kernel.unsqueeze(0).unsqueeze(0), padding=padding)
            channels.append(blurred.squeeze(0))
        
        return torch.cat(channels, dim=0)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel for blurring"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel = g.unsqueeze(1) * g.unsqueeze(0)
        return kernel
    
    def _underwater_distortion(
        self, 
        image: torch.Tensor, 
        strength: float
    ) -> torch.Tensor:
        """
        Apply underwater-specific distortions
        
        Simulates:
        - Water column effects
        - Particle scattering
        - Color attenuation with depth
        """
        img = image.clone()
        height, width = img.shape[1:]
        
        # 1. Blue-green color shift (water filtering)
        color_shift = torch.tensor([0.9, 0.95, 1.1]).to(img.device).view(3, 1, 1)
        img = img * (1 - strength * 0.3 + strength * 0.3 * color_shift)
        
        # 2. Vignetting effect (light attenuation)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        y, x = y.to(img.device), x.to(img.device)
        
        # Distance from center
        distance = torch.sqrt(x**2 + y**2)
        vignette = 1 - strength * 0.5 * (distance / distance.max())**2
        vignette = vignette.clamp(0.3, 1.0)
        
        img = img * vignette.unsqueeze(0)
        
        # 3. Slight haze effect
        haze = torch.ones_like(img) * 0.1 * strength
        img = img * (1 - strength * 0.2) + haze
        
        return torch.clamp(img, 0, 1)
    
    def _lighting_change(
        self,
        image: torch.Tensor,
        direction: str,
        strength: float
    ) -> torch.Tensor:
        """
        Apply directional lighting changes
        
        Simulates changing light conditions underwater
        """
        img = image.clone()
        height, width = img.shape[1:]
        
        # Create lighting gradient
        if direction == 'top':
            gradient = torch.linspace(1 + strength, 1 - strength, height)
            gradient = gradient.view(-1, 1).expand(height, width)
        elif direction == 'bottom':
            gradient = torch.linspace(1 - strength, 1 + strength, height)
            gradient = gradient.view(-1, 1).expand(height, width)
        elif direction == 'left':
            gradient = torch.linspace(1 + strength, 1 - strength, width)
            gradient = gradient.view(1, -1).expand(height, width)
        elif direction == 'right':
            gradient = torch.linspace(1 - strength, 1 + strength, width)
            gradient = gradient.view(1, -1).expand(height, width)
        
        gradient = gradient.to(img.device).unsqueeze(0)
        img = img * gradient
        
        return torch.clamp(img, 0, 1)


class SequenceConsistentAugmentation:
    """
    Augmentation that maintains consistency across temporal sequences
    
    Some augmentations should vary across time (like lighting changes)
    while others should remain consistent (like camera parameters)
    """
    
    def __init__(self, base_augmentation: UnderwaterAugmentation):
        self.base_aug = base_augmentation
        
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply sequence-aware augmentations
        
        Args:
            images: Input images [seq_len, num_cameras, 3, H, W]
        Returns:
            Augmented images [seq_len, num_cameras, 3, H, W]
        """
        seq_len, num_cameras = images.shape[:2]
        
        # Sample consistent parameters for the entire sequence
        consistent_params = self.base_aug._sample_augmentation_parameters()
        
        # Apply time-varying augmentations
        augmented_sequence = []
        
        for t in range(seq_len):
            # Some parameters can vary with time
            time_varying_params = consistent_params.copy()
            
            # Gradually change lighting (simulate movement through water)
            if consistent_params.get('apply_lighting_change', False):
                time_factor = t / max(seq_len - 1, 1)
                time_varying_params['lighting_strength'] *= (0.5 + 0.5 * time_factor)
            
            # Apply augmentations to all cameras at this time step
            frame_cameras = []
            for c in range(num_cameras):
                img = images[t, c]
                aug_img = self.base_aug._apply_augmentations(img, time_varying_params)
                frame_cameras.append(aug_img)
            
            augmented_sequence.append(torch.stack(frame_cameras))
        
        return torch.stack(augmented_sequence)