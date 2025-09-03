import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json

# Try to import torch and numpy, but make them optional for basic testing
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


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary for testing."""
    return {
        "method": "inflora",
        "dataset": "cifar100",
        "num_classes": 100,
        "increment": 10,
        "device": "cpu",
        "epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "memory_size": 2000,
        "seed": 42
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """Create a temporary config file for testing."""
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_device():
    """Mock torch device for testing."""
    if TORCH_AVAILABLE:
        return torch.device('cpu')
    return Mock()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    if TORCH_AVAILABLE:
        return torch.randn(4, 3, 32, 32)
    # Return a mock tensor-like object if torch is not available
    mock_tensor = Mock()
    mock_tensor.shape = (4, 3, 32, 32)
    return mock_tensor


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    if TORCH_AVAILABLE:
        return torch.randint(0, 10, (4,))
    # Return a mock tensor-like object if torch is not available
    mock_labels = Mock()
    mock_labels.shape = (4,)
    return mock_labels


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    if NUMPY_AVAILABLE:
        return np.random.randn(4, 3, 32, 32).astype(np.float32)
    # Return a mock array-like object if numpy is not available
    mock_array = Mock()
    mock_array.shape = (4, 3, 32, 32)
    mock_array.dtype = 'float32'
    return mock_array


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval.return_value = model
    model.train.return_value = model
    
    if TORCH_AVAILABLE:
        model.forward.return_value = torch.randn(4, 10)
        model.state_dict.return_value = {"layer1.weight": torch.randn(10, 5)}
        model.parameters.return_value = [torch.randn(10, 5, requires_grad=True)]
    else:
        model.forward.return_value = Mock()
        model.state_dict.return_value = {"layer1.weight": Mock()}
        model.parameters.return_value = [Mock()]
    
    model.load_state_dict = Mock()
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.state_dict.return_value = {"state": {}}
    optimizer.load_state_dict = Mock()
    return optimizer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = Mock()
    if TORCH_AVAILABLE:
        sample_batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
    else:
        sample_batch = (Mock(), Mock())
    dataloader.__iter__.return_value = iter([sample_batch, sample_batch])
    dataloader.__len__.return_value = 2
    return dataloader


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__.return_value = 100
    if TORCH_AVAILABLE:
        dataset.__getitem__.return_value = (torch.randn(3, 32, 32), torch.tensor(5))
    else:
        dataset.__getitem__.return_value = (Mock(), Mock())
    return dataset


@pytest.fixture
def sample_task_data():
    """Create sample task data structure for continual learning."""
    return {
        "task_id": 0,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "train_loader": Mock(),
        "test_loader": Mock(),
        "num_samples": 5000
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing."""
    wandb_mock = Mock()
    wandb_mock.init = Mock()
    wandb_mock.log = Mock()
    wandb_mock.finish = Mock()
    wandb_mock.watch = Mock()
    return wandb_mock


@pytest.fixture
def set_random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    if NUMPY_AVAILABLE:
        np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    yield seed


@pytest.fixture
def memory_buffer_data():
    """Create sample memory buffer data."""
    if TORCH_AVAILABLE:
        return {
            "images": torch.randn(100, 3, 32, 32),
            "labels": torch.randint(0, 10, (100,)),
            "task_ids": torch.randint(0, 3, (100,))
        }
    else:
        mock_images = Mock()
        mock_images.shape = (100, 3, 32, 32)
        mock_labels = Mock()
        mock_labels.shape = (100,)
        mock_task_ids = Mock()
        mock_task_ids.shape = (100,)
        return {
            "images": mock_images,
            "labels": mock_labels,
            "task_ids": mock_task_ids
        }


@pytest.fixture
def sample_lora_config():
    """Create sample LoRA configuration."""
    return {
        "rank": 4,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["query", "key", "value"],
        "bias": "none"
    }


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Automatically cleanup CUDA cache after each test."""
    yield
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint for testing."""
    if TORCH_AVAILABLE:
        model_state_dict = {"layer1.weight": torch.randn(10, 5)}
    else:
        model_state_dict = {"layer1.weight": Mock()}
    
    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": {"state": {}},
        "epoch": 10,
        "loss": 0.5,
        "accuracy": 0.85,
        "task_id": 1
    }


@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing.""" 
    return {
        "accuracy": 0.85,
        "loss": 0.25,
        "precision": 0.83,
        "recall": 0.87,
        "f1_score": 0.85,
        "forgetting": 0.05
    }