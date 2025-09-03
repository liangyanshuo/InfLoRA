"""
Validation tests to ensure the testing infrastructure is set up correctly.
"""

import pytest
from pathlib import Path
import tempfile
import json

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_pytest_basic_functionality(self):
        """Test that pytest is working correctly."""
        assert True
        assert 1 + 1 == 2
        assert "hello" == "hello"

    def test_fixtures_are_working(self, sample_config, temp_dir):
        """Test that fixtures are loaded and working."""
        assert isinstance(sample_config, dict)
        assert "method" in sample_config
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()

    def test_torch_functionality(self, sample_tensor, sample_labels):
        """Test that PyTorch is working correctly."""
        if TORCH_AVAILABLE:
            assert torch.is_tensor(sample_tensor)
            assert torch.is_tensor(sample_labels)
        assert sample_tensor.shape == (4, 3, 32, 32)
        assert sample_labels.shape == (4,)

    def test_numpy_functionality(self, sample_numpy_array):
        """Test that NumPy is working correctly."""
        if NUMPY_AVAILABLE:
            assert isinstance(sample_numpy_array, np.ndarray)
            assert sample_numpy_array.dtype == np.float32
        assert sample_numpy_array.shape == (4, 3, 32, 32)

    def test_mock_functionality(self, mock_model, mock_optimizer):
        """Test that mocking is working correctly."""
        assert mock_model is not None
        assert mock_optimizer is not None
        
        # Test mock calls
        mock_model.eval()
        mock_model.eval.assert_called_once()
        
        mock_optimizer.zero_grad()
        mock_optimizer.zero_grad.assert_called_once()

    def test_temporary_file_creation(self, temp_dir):
        """Test that temporary files can be created and accessed."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, testing!")
        
        assert test_file.exists()
        assert test_file.read_text() == "Hello, testing!"

    def test_config_file_fixture(self, sample_config_file, sample_config):
        """Test that config file fixture works correctly."""
        assert sample_config_file.exists()
        
        with open(sample_config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == sample_config

    def test_memory_buffer_fixture(self, memory_buffer_data):
        """Test that memory buffer fixture is working."""
        assert "images" in memory_buffer_data
        assert "labels" in memory_buffer_data
        assert "task_ids" in memory_buffer_data
        
        assert memory_buffer_data["images"].shape == (100, 3, 32, 32)
        assert memory_buffer_data["labels"].shape == (100,)
        assert memory_buffer_data["task_ids"].shape == (100,)

    def test_lora_config_fixture(self, sample_lora_config):
        """Test that LoRA configuration fixture is working."""
        required_keys = ["rank", "alpha", "dropout", "target_modules", "bias"]
        for key in required_keys:
            assert key in sample_lora_config
        
        assert isinstance(sample_lora_config["rank"], int)
        assert isinstance(sample_lora_config["alpha"], int)
        assert isinstance(sample_lora_config["dropout"], float)
        assert isinstance(sample_lora_config["target_modules"], list)

    def test_metrics_fixture(self, sample_metrics):
        """Test that metrics fixture is working."""
        expected_metrics = ["accuracy", "loss", "precision", "recall", "f1_score", "forgetting"]
        for metric in expected_metrics:
            assert metric in sample_metrics
            assert isinstance(sample_metrics[metric], float)

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker is working."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker is working."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is working."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True

    def test_random_seed_fixture(self, set_random_seed):
        """Test that random seed fixture provides reproducible results."""
        assert set_random_seed == 42
        
        # Test reproducible random numbers if libraries are available
        if TORCH_AVAILABLE:
            torch_random = torch.randn(3, 3)
            assert torch_random.shape == (3, 3)
            
        if NUMPY_AVAILABLE:
            numpy_random = np.random.randn(3, 3)
            assert numpy_random.shape == (3, 3)

    def test_task_data_fixture(self, sample_task_data):
        """Test that task data fixture is working."""
        assert "task_id" in sample_task_data
        assert "classes" in sample_task_data
        assert "train_loader" in sample_task_data
        assert "test_loader" in sample_task_data
        assert "num_samples" in sample_task_data
        
        assert sample_task_data["task_id"] == 0
        assert len(sample_task_data["classes"]) == 10

    def test_checkpoint_fixture(self, mock_checkpoint):
        """Test that checkpoint fixture is working."""
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "loss", "accuracy", "task_id"]
        for key in required_keys:
            assert key in mock_checkpoint
        
        assert isinstance(mock_checkpoint["epoch"], int)
        assert isinstance(mock_checkpoint["loss"], float)
        assert isinstance(mock_checkpoint["accuracy"], float)


class TestInfrastructureComponents:
    """Test infrastructure components are properly configured."""

    def test_imports_work(self):
        """Test that all necessary imports work."""
        import pytest
        import tempfile
        import pathlib
        import json
        from unittest.mock import Mock, MagicMock
        
        assert pytest is not None

    def test_torch_device_availability(self, mock_device):
        """Test torch device configuration."""
        if TORCH_AVAILABLE:
            assert mock_device.type == 'cpu'
            # Test that CUDA info can be accessed even if not available
            cuda_available = torch.cuda.is_available()
            assert isinstance(cuda_available, bool)
        else:
            # Just check that the mock device exists
            assert mock_device is not None

    def test_directory_structure_exists(self):
        """Test that the test directory structure was created correctly."""
        test_dir = Path(__file__).parent
        
        assert test_dir.name == "tests"
        assert (test_dir / "__init__.py").exists()
        assert (test_dir / "conftest.py").exists()
        assert (test_dir / "unit").exists()
        assert (test_dir / "integration").exists()
        assert (test_dir / "unit" / "__init__.py").exists()
        assert (test_dir / "integration" / "__init__.py").exists()