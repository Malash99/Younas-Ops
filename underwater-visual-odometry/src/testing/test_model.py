import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir.parent))

try:
    from models.underwater_vio_lstm import UnderwaterVIODataset, create_model
    from models.vio_losses import create_loss_function
except ImportError:
    try:
        from src.models.underwater_vio_lstm import UnderwaterVIODataset, create_model
        from src.models.vio_losses import create_loss_function
    except ImportError as e:
        print(f"Import error: {e}")
        raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTester:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model().to(self.device)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("Using randomly initialized model for testing")
        
        self.model.eval()
    
    def test_data_loading(self, csv_path, data_root):
        """Test dataset loading functionality"""
        logger.info("Testing dataset loading...")
        
        try:
            dataset = UnderwaterVIODataset(
                csv_path=csv_path,
                data_root=data_root,
                sequence_length=10,
                use_ground_truth_only=True
            )
            
            logger.info(f"Dataset size: {len(dataset)}")
            
            # Test a few samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                logger.info(f"Sample {i}:")
                logger.info(f"  Images shape: {sample['images'].shape}")
                logger.info(f"  IMU shape: {sample['imu'].shape}")
                logger.info(f"  Pose shape: {sample['pose'].shape}")
                logger.info(f"  Timestamps shape: {sample['timestamps'].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            return False
    
    def test_model_forward_pass(self, csv_path, data_root):
        """Test model forward pass with real data"""
        logger.info("Testing model forward pass...")
        
        try:
            dataset = UnderwaterVIODataset(
                csv_path=csv_path,
                data_root=data_root,
                sequence_length=10,
                use_ground_truth_only=True
            )
            
            if len(dataset) == 0:
                logger.warning("No samples in dataset, using dummy data")
                return self.test_with_dummy_data()
            
            # Get a sample
            sample = dataset[0]
            images = sample['images'].unsqueeze(0).to(self.device)
            imu = sample['imu'].unsqueeze(0).to(self.device)
            pose = sample['pose'].unsqueeze(0).to(self.device)
            
            logger.info(f"Input shapes - Images: {images.shape}, IMU: {imu.shape}")
            
            with torch.no_grad():
                predictions = self.model(images, imu)
                logger.info(f"Output shape: {predictions.shape}")
                logger.info(f"Predictions sample: {predictions[0, :3].cpu().numpy()}")
                logger.info(f"Targets sample: {pose[0, :3].cpu().numpy()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return False
    
    def test_with_dummy_data(self):
        """Test model with dummy data"""
        logger.info("Testing with dummy data...")
        
        batch_size, seq_len = 2, 10
        images = torch.randn(batch_size, seq_len, 3, 224, 224).to(self.device)
        imu = torch.randn(batch_size, seq_len, 6).to(self.device)
        
        try:
            with torch.no_grad():
                predictions = self.model(images, imu)
                logger.info(f"Dummy test successful! Output shape: {predictions.shape}")
                return True
        except Exception as e:
            logger.error(f"Dummy test failed: {e}")
            return False
    
    def test_constant_prediction_detection(self, csv_path, data_root, num_samples=50):
        """Test if model produces constant predictions"""
        logger.info("Testing for constant predictions...")
        
        try:
            dataset = UnderwaterVIODataset(
                csv_path=csv_path,
                data_root=data_root,
                sequence_length=10,
                use_ground_truth_only=True
            )
            
            if len(dataset) == 0:
                logger.warning("No dataset available for constant prediction test")
                return self.test_constant_prediction_dummy()
            
            predictions_list = []
            num_samples = min(num_samples, len(dataset))
            
            with torch.no_grad():
                for i in range(num_samples):
                    sample = dataset[i]
                    images = sample['images'].unsqueeze(0).to(self.device)
                    imu = sample['imu'].unsqueeze(0).to(self.device)
                    
                    predictions = self.model(images, imu)
                    predictions_list.append(predictions.cpu().numpy())
            
            # Analyze predictions
            all_predictions = np.concatenate(predictions_list, axis=0)  # (num_samples, seq_len, 6)
            
            # Compute statistics
            pred_means = np.mean(all_predictions, axis=(0, 1))  # Mean across samples and time
            pred_stds = np.std(all_predictions, axis=(0, 1))    # Std across samples and time
            
            # Check for constant predictions
            pose_names = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
            
            logger.info("Prediction analysis:")
            constant_detected = False
            for i, name in enumerate(pose_names):
                logger.info(f"{name}: mean={pred_means[i]:.6f}, std={pred_stds[i]:.6f}")
                if pred_stds[i] < 1e-6:  # Very small standard deviation
                    logger.warning(f"CONSTANT PREDICTION DETECTED for {name}!")
                    constant_detected = True
            
            # Additional analysis: check variance within sequences
            within_seq_vars = np.var(all_predictions, axis=1)  # Variance within each sequence
            mean_within_seq_var = np.mean(within_seq_vars, axis=0)
            
            logger.info("Within-sequence variance:")
            for i, name in enumerate(pose_names):
                logger.info(f"{name}: {mean_within_seq_var[i]:.8f}")
                if mean_within_seq_var[i] < 1e-8:
                    logger.warning(f"LOW WITHIN-SEQUENCE VARIANCE for {name}!")
                    constant_detected = True
            
            if not constant_detected:
                logger.info("✓ No constant prediction issues detected")
            
            return not constant_detected
            
        except Exception as e:
            logger.error(f"Constant prediction test failed: {e}")
            return False
    
    def test_constant_prediction_dummy(self):
        """Test constant prediction with dummy data"""
        logger.info("Testing constant predictions with dummy data...")
        
        batch_size, seq_len = 10, 10
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(20):  # Test with multiple batches
                images = torch.randn(batch_size, seq_len, 3, 224, 224).to(self.device)
                imu = torch.randn(batch_size, seq_len, 6).to(self.device)
                
                predictions = self.model(images, imu)
                predictions_list.append(predictions.cpu().numpy())
        
        # Analyze
        all_predictions = np.concatenate(predictions_list, axis=0)
        pred_stds = np.std(all_predictions, axis=(0, 1))
        
        pose_names = ['delta_x', 'delta_y', 'delta_z', 'delta_roll', 'delta_pitch', 'delta_yaw']
        constant_detected = False
        
        for i, name in enumerate(pose_names):
            logger.info(f"{name}: std={pred_stds[i]:.8f}")
            if pred_stds[i] < 1e-6:
                logger.warning(f"CONSTANT PREDICTION DETECTED for {name}!")
                constant_detected = True
        
        return not constant_detected
    
    def test_loss_functions(self):
        """Test loss functions"""
        logger.info("Testing loss functions...")
        
        try:
            criterion = create_loss_function()
            
            # Create test data
            batch_size, seq_len = 4, 10
            pred_poses = torch.randn(batch_size, seq_len, 6)
            target_poses = torch.randn(batch_size, seq_len, 6)
            images = torch.randn(batch_size, seq_len, 3, 224, 224)
            
            # Test loss computation
            loss, loss_dict = criterion(pred_poses, target_poses, images)
            
            logger.info("Loss function test successful!")
            logger.info(f"Total loss: {loss.item():.6f}")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"{key}: {value.item():.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Loss function test failed: {e}")
            return False
    
    def run_all_tests(self, csv_path, data_root):
        """Run all tests"""
        logger.info("Running comprehensive model tests...")
        
        tests = [
            ("Data Loading", lambda: self.test_data_loading(csv_path, data_root)),
            ("Model Forward Pass", lambda: self.test_model_forward_pass(csv_path, data_root)),
            ("Dummy Data Test", self.test_with_dummy_data),
            ("Loss Functions", self.test_loss_functions),
            ("Constant Prediction Detection", lambda: self.test_constant_prediction_detection(csv_path, data_root))
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = result
                status = "✓ PASSED" if result else "✗ FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✓" if passed_test else "✗"
            logger.info(f"{status} {test_name}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        return results


def main():
    # Configuration
    csv_path = r'data\processed\visual_odometry_dataset\visual_odometry_dataset_kalibr_clean.csv'
    data_root = r'data\processed\visual_odometry_dataset'
    model_path = None  # Set to checkpoint path if available
    
    # Create tester and run tests
    tester = ModelTester(model_path=model_path)
    results = tester.run_all_tests(csv_path, data_root)
    
    # Return exit code based on results
    if all(results.values()):
        logger.info("All tests passed! ✓")
        return 0
    else:
        logger.error("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    exit(main())