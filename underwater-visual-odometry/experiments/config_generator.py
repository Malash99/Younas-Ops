"""
Configuration generator for all UW-TransVO experiments

Generates configs for the complete experimental matrix:
- Camera configurations: [0], [0,1], [0,1,2], [0,1,2,3]
- Sequence lengths: 2, 4
- Modalities: vision, +imu, +pressure, +both
"""

from typing import Dict, List
import itertools
import json
from pathlib import Path


class ExperimentConfigGenerator:
    """Generate experiment configurations for comprehensive evaluation"""
    
    def __init__(self, base_config: Dict):
        """
        Args:
            base_config: Base configuration dictionary
        """
        self.base_config = base_config
        
    def generate_all_configs(self) -> List[Dict]:
        """Generate all experiment configurations"""
        
        # Define experimental variables
        camera_configs = {
            'mono': [0],
            'stereo': [0, 2], 
            'triple': [0, 1, 2],
            'quad': [0, 1, 2, 3]
        }
        
        sequence_lengths = [2, 4]
        
        modality_configs = {
            'vision': {'use_imu': False, 'use_pressure': False},
            'vision_imu': {'use_imu': True, 'use_pressure': False},
            'vision_pressure': {'use_imu': False, 'use_pressure': True},
            'vision_all': {'use_imu': True, 'use_pressure': True}
        }
        
        # Generate all combinations
        configs = []
        
        for (cam_name, cameras), seq_len, (mod_name, modality) in itertools.product(
            camera_configs.items(), 
            sequence_lengths, 
            modality_configs.items()
        ):
            # Create experiment name
            exp_name = f"{cam_name}_seq{seq_len}_{mod_name}"
            
            # Create config
            config = self._create_config(
                experiment_name=exp_name,
                cameras=cameras,
                sequence_length=seq_len,
                modality=modality
            )
            
            configs.append(config)
        
        return configs
    
    def generate_ablation_configs(self) -> List[Dict]:
        """Generate configurations for ablation studies"""
        
        ablation_configs = []
        
        # Base configuration (quad cameras, sequence 2, all modalities)
        base_cameras = [0, 1, 2, 3]
        base_seq_len = 2
        base_modality = {'use_imu': True, 'use_pressure': True}
        
        # 1. Ablation: Remove attention components
        attention_ablations = [
            ('no_spatial_attention', {'disable_spatial_attention': True}),
            ('no_temporal_attention', {'disable_temporal_attention': True}),
            ('no_cross_modal_attention', {'disable_cross_modal_attention': True})
        ]
        
        for ablation_name, ablation_params in attention_ablations:
            config = self._create_config(
                experiment_name=f"ablation_{ablation_name}",
                cameras=base_cameras,
                sequence_length=base_seq_len,
                modality=base_modality,
                **ablation_params
            )
            ablation_configs.append(config)
        
        # 2. Ablation: Different loss functions
        loss_ablations = [
            ('mse_loss', {'loss': {'loss_type': 'pose', 'base_loss': 'mse'}}),
            ('huber_loss', {'loss': {'loss_type': 'robust', 'robust_type': 'huber'}}),
            ('uncertainty_loss', {'loss': {'loss_type': 'uncertainty'}})
        ]
        
        for ablation_name, ablation_params in loss_ablations:
            config = self._create_config(
                experiment_name=f"ablation_{ablation_name}",
                cameras=base_cameras,
                sequence_length=base_seq_len,
                modality=base_modality
            )
            config.update(ablation_params)
            ablation_configs.append(config)
        
        # 3. Ablation: Different model sizes
        model_size_ablations = [
            ('tiny', {'model': {'d_model': 192, 'num_heads': 3, 'num_layers': 6}}),
            ('small', {'model': {'d_model': 384, 'num_heads': 6, 'num_layers': 8}}),
            ('large', {'model': {'d_model': 1024, 'num_heads': 16, 'num_layers': 12}})
        ]
        
        for ablation_name, ablation_params in model_size_ablations:
            config = self._create_config(
                experiment_name=f"ablation_model_{ablation_name}",
                cameras=base_cameras,
                sequence_length=base_seq_len,
                modality=base_modality
            )
            config.update(ablation_params)
            ablation_configs.append(config)
        
        return ablation_configs
    
    def _create_config(
        self,
        experiment_name: str,
        cameras: List[int],
        sequence_length: int,
        modality: Dict,
        **kwargs
    ) -> Dict:
        """Create a single experiment configuration"""
        
        # Start with base config
        config = self.base_config.copy()
        
        # Update experiment-specific settings
        config.update({
            'experiment_name': experiment_name,
            'data': {
                **config.get('data', {}),
                'camera_ids': cameras,
                'sequence_length': sequence_length,
                **modality
            },
            'model': {
                **config.get('model', {}),
                'max_cameras': len(cameras),
                'max_seq_len': sequence_length,
                **modality
            }
        })
        
        # Apply any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        
        return config


def generate_experiment_configs(base_config_path: str) -> List[Dict]:
    """
    Generate all experiment configurations
    
    Args:
        base_config_path: Path to base configuration file
        
    Returns:
        List of experiment configurations
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    generator = ExperimentConfigGenerator(base_config)
    
    # Generate main experiments
    main_configs = generator.generate_all_configs()
    
    # Generate ablation studies
    ablation_configs = generator.generate_ablation_configs()
    
    # Combine all configs
    all_configs = main_configs + ablation_configs
    
    print(f"Generated {len(main_configs)} main experiment configs")
    print(f"Generated {len(ablation_configs)} ablation study configs")
    print(f"Total: {len(all_configs)} experiment configurations")
    
    return all_configs


def save_experiment_configs(configs: List[Dict], output_dir: str):
    """Save experiment configurations to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual configs
    configs_dir = output_path / 'configs'
    configs_dir.mkdir(exist_ok=True)
    
    for config in configs:
        exp_name = config['experiment_name']
        config_file = configs_dir / f"{exp_name}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Save summary
    summary = {
        'total_experiments': len(configs),
        'experiment_names': [config['experiment_name'] for config in configs],
        'camera_configurations': list(set(str(config['data']['camera_ids']) for config in configs)),
        'sequence_lengths': list(set(config['data']['sequence_length'] for config in configs)),
        'modalities': list(set(f"imu_{config['data']['use_imu']}_pressure_{config['data']['use_pressure']}" for config in configs))
    }
    
    summary_file = output_path / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved {len(configs)} experiment configs to {output_dir}")
    print(f"Summary saved to {summary_file}")


if __name__ == '__main__':
    # Example usage
    base_config = {
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 6,
            'dropout': 0.1,
            'uncertainty_estimation': True
        },
        'data': {
            'data_csv': 'data/processed/full_dataset/data/training_data.csv',
            'data_root': 'data/processed/full_dataset',
            'img_size': 224,
            'augmentation': True,
            'max_samples': None
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'log_interval': 50
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6
        },
        'loss': {
            'loss_type': 'pose',
            'translation_weight': 1.0,
            'rotation_weight': 10.0,
            'base_loss': 'mse'
        }
    }
    
    generator = ExperimentConfigGenerator(base_config)
    configs = generator.generate_all_configs()
    
    save_experiment_configs(configs, 'experiments/configs')