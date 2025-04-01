from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Generic, TypeVar, List, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Forward reference for ProbeVector to avoid circular imports
ProbeVector = None

@dataclass
class ProbeConfig:
    """Base configuration for probes with metadata."""
    # Core configuration
    input_size: int
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Metadata fields (from former ProbeVector)
    model_name: str = "unknown_model"
    hook_point: str = "unknown_hook"
    hook_layer: int = 0
    hook_head_index: Optional[int] = None
    name: str = "unnamed_probe"
    
    # Dataset information
    dataset_path: Optional[str] = None
    prepend_bos: bool = True
    context_size: int = 128
    
    # Technical settings
    dtype: str = "float32"
    
    # Additional metadata
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LinearProbeConfig(ProbeConfig):
    """Configuration for linear probe."""
    loss_type: Literal["mse", "cosine", "l1"] = "mse"
    normalize_weights: bool = True
    bias: bool = False
    output_size: int = 1  # Number of output dimensions

@dataclass
class LogisticProbeConfig(ProbeConfig):
    """Configuration for logistic regression probe."""
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1  # Number of output dimensions

@dataclass
class KMeansProbeConfig(ProbeConfig):
    """Configuration for K-means clustering probe."""
    n_clusters: int = 2
    n_init: int = 10
    normalize_weights: bool = True
    random_state: int = 42

@dataclass
class PCAProbeConfig(ProbeConfig):
    """Configuration for PCA-based probe."""
    n_components: int = 1
    normalize_weights: bool = True

@dataclass
class MeanDiffProbeConfig(ProbeConfig):
    """Configuration for mean difference probe."""
    normalize_weights: bool = True

T = TypeVar('T', bound=ProbeConfig)


class BaseProbe(ABC, nn.Module, Generic[T]):
    """Abstract base class for probes with vector functionality."""
    
    def __init__(self, config: T):
        super().__init__()
        self.config = config
        self.dtype = torch.float32  # Add default dtype
        self.name = config.name or "unnamed_probe"  # Use config name or default
        
    @abstractmethod
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction."""
        pass
    
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot product between activations and the probe direction vector."""
        direction = self.get_direction()
        return torch.einsum("...d,d->...", acts, direction)
    
    def normalize_vector(self) -> None:
        """Normalize the probe direction to unit length."""
        direction = self.get_direction()
        if hasattr(self, "linear"):
            with torch.no_grad():
                self.linear.weight.data = torch.nn.functional.normalize(self.linear.weight.data, p=2, dim=1)
        elif hasattr(self, "direction"):
            self.direction = torch.nn.functional.normalize(self.direction, p=2, dim=0)
        
    def save(self, path: str) -> None:
        """Save probe state, config, and direction in a single file."""
        # Get direction
        direction = self.get_direction()
        
        # Get additional info including standardization buffers
        additional_info = dict(self.config.additional_info)
        if hasattr(self, 'feature_mean') and self.feature_mean is not None:
            additional_info['feature_mean'] = self.feature_mean.cpu().numpy().tolist()
        if hasattr(self, 'feature_std') and self.feature_std is not None:
            additional_info['feature_std'] = self.feature_std.cpu().numpy().tolist()
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            additional_info['bias'] = self.linear.bias.data.cpu().numpy().tolist()
            
        # Update config with latest additional_info
        self.config.additional_info = additional_info
        
        # Save full state
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'direction': direction.cpu(),
            'probe_type': self.__class__.__name__
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BaseProbe':
        """Load probe from saved state with enhanced compatibility."""
        if path.endswith(".json"):
            # Use load_json if it's a JSON file
            return cls.load_json(path)
            
        # Load data
        data = torch.load(path, weights_only=False)
        
        # Create probe with the saved config
        probe = cls(data['config'])
        
        # Load state dict if available
        if 'state_dict' in data:
            probe.load_state_dict(data['state_dict'])
        # Otherwise set direction directly
        elif 'direction' in data:
            direction = data['direction']
            if hasattr(probe, "linear"):
                with torch.no_grad():
                    probe.linear.weight.data = direction.unsqueeze(0)
            else:
                probe.direction = direction
        
        # Set the probe name
        probe.name = getattr(data['config'], 'name', "unnamed_probe")
        
        return probe

    def save_json(self, path: str) -> None:
        """Save probe direction and metadata as JSON."""
        # ensure the path ends in .json
        if not path.endswith(".json"):
            path += ".json"

        # ensure folder exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get probe direction, but without normalizing or rescaling
        # This gets the raw linear weights so we don't apply transforms twice
        raw_weight = self.linear.weight.data.squeeze(0).cpu()
        
        # Prepare metadata
        metadata = {
            "model_name": self.config.model_name,
            "hook_point": self.config.hook_point,
            "hook_layer": self.config.hook_layer,
            "hook_head_index": self.config.hook_head_index,
            "vector_name": self.name,
            "vector_dimension": raw_weight.shape[0],
            "probe_type": self.__class__.__name__,
            "dataset_path": self.config.dataset_path,
            "prepend_bos": self.config.prepend_bos,
            "context_size": self.config.context_size,
            "dtype": self.config.dtype,
            "device": self.config.device,
        }
        
        # Get additional info including standardization buffers
        additional_info = dict(self.config.additional_info)
        if hasattr(self, 'feature_mean') and self.feature_mean is not None:
            additional_info['feature_mean'] = self.feature_mean.cpu().numpy().tolist()
        if hasattr(self, 'feature_std') and self.feature_std is not None:
            additional_info['feature_std'] = self.feature_std.cpu().numpy().tolist()
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            additional_info['bias'] = self.linear.bias.data.cpu().numpy().tolist()
        
        # Mark that this vector has already been normalized and unscaled
        # This will ensure we don't reapply transformations when loading
        additional_info['is_normalized'] = True
        additional_info['is_unscaled'] = True
        
        metadata["additional_info"] = additional_info
        
        # Save as JSON
        save_data = {
            "vector": raw_weight.numpy().tolist(),
            "metadata": metadata
        }
        
        with open(path, "w") as f:
            json.dump(save_data, f)
            
    @classmethod
    def load_json(cls, path: str, config: Optional[T] = None) -> 'BaseProbe':
        """Load probe from JSON file with enhanced compatibility."""
        with open(path, "r") as f:
            data = json.load(f)
            
        # Determine format and extract data
        if "vector" in data and "metadata" in data:
            # New JSON format
            vector = torch.tensor(data["vector"])
            metadata = data["metadata"]
            
            # Create config if not provided
            if config is None:
                dim = vector.shape[0]
                
                # Create appropriate config based on class
                if cls.__name__ == "LinearProbe":
                    config = LinearProbeConfig(input_size=dim)
                elif cls.__name__ == "LogisticProbe":
                    config = LogisticProbeConfig(input_size=dim)
                elif cls.__name__ == "KMeansProbe":
                    config = KMeansProbeConfig(input_size=dim)
                elif cls.__name__ == "PCAProbe":
                    config = PCAProbeConfig(input_size=dim)
                elif cls.__name__ == "MeanDifferenceProbe":
                    config = MeanDiffProbeConfig(input_size=dim)
                else:
                    config = ProbeConfig(input_size=dim)
                
                # Update config with metadata
                config.model_name = metadata.get("model_name", "unknown_model")
                config.hook_point = metadata.get("hook_point", "unknown_hook")
                config.hook_layer = metadata.get("hook_layer", 0)
                config.hook_head_index = metadata.get("hook_head_index")
                config.name = metadata.get("vector_name", "unnamed_probe")
                config.dataset_path = metadata.get("dataset_path")
                config.prepend_bos = metadata.get("prepend_bos", True)
                config.context_size = metadata.get("context_size", 128)
                config.dtype = metadata.get("dtype", "float32")
                config.device = metadata.get("device", "cpu")
                config.additional_info = metadata.get("additional_info", {})
                
                # Add information about whether the vector is already pre-processed
                if "is_normalized" not in config.additional_info:
                    config.additional_info["is_normalized"] = True
                    config.additional_info["is_unscaled"] = True
        else:
            # Legacy format
            if "vectors" in data and isinstance(data["vectors"], list):
                vector = torch.tensor(data["vectors"][0])
            elif "direction" in data:
                vector = torch.tensor(data["direction"])
            elif isinstance(data, list):
                vector = torch.tensor(data)
            else:
                raise ValueError(f"Unrecognized JSON format in {path}")
                
            # Create a basic config if not provided
            if config is None:
                dim = vector.shape[0]
                config = ProbeConfig(input_size=dim)
        
        # Create the probe
        probe = cls(config)
        
        # Set the vector direction
        if hasattr(probe, "linear"):
            with torch.no_grad():
                # Unsqueeze and set as weight
                probe.linear.weight.data = vector.unsqueeze(0)
                
                # Restore bias if it exists
                if (hasattr(probe.linear, "bias") and probe.linear.bias is not None and 
                    "additional_info" in metadata and "bias" in metadata["additional_info"]):
                    bias_data = torch.tensor(metadata["additional_info"]["bias"])
                    probe.linear.bias.data = bias_data
        else:
            probe.direction = vector
        
        # Restore feature mean and std if they exist
        if "additional_info" in metadata:
            if "feature_mean" in metadata["additional_info"]:
                mean_data = torch.tensor(metadata["additional_info"]["feature_mean"])
                probe.register_buffer("feature_mean", mean_data)
                
            if "feature_std" in metadata["additional_info"]:
                std_data = torch.tensor(metadata["additional_info"]["feature_std"])
                probe.register_buffer("feature_std", std_data)
        
        # Set the probe name
        probe.name = config.name
        
        return probe
    
    # Backward compatibility
    def to_probe_vector(
        self, 
        model_name: str = None,
        hook_point: str = None,
        hook_layer: int = None,
        **kwargs
    ):
        """For backward compatibility - stores metadata and returns self."""
        # Update config with provided values or keep existing ones
        self.config.model_name = model_name or self.config.model_name
        self.config.hook_point = hook_point or self.config.hook_point
        self.config.hook_layer = hook_layer if hook_layer is not None else self.config.hook_layer
        
        # Get name
        if 'name' in kwargs:
            self.name = kwargs['name']
            self.config.name = kwargs['name']
            
        # Update additional info
        if 'additional_info' in kwargs:
            self.config.additional_info.update(kwargs['additional_info'])
            
        # For backward compatibility, add feature_mean and feature_std to additional_info
        if hasattr(self, 'feature_mean') and self.feature_mean is not None:
            self.config.additional_info['feature_mean'] = self.feature_mean.cpu().numpy().tolist()
        if hasattr(self, 'feature_std') and self.feature_std is not None:
            self.config.additional_info['feature_std'] = self.feature_std.cpu().numpy().tolist()
            
        # For backward compatibility, return self to enable chaining
        return self


class ProbeSet:
    """A collection of probes."""
    
    def __init__(
        self,
        probes: List[BaseProbe],
    ):
        self.probes = probes
        
        # Validate that all probes have compatible dimensions
        dims = [p.get_direction().shape[0] for p in probes]
        if len(set(dims)) > 1:
            raise ValueError(f"All probes must have the same input dimension, got {dims}")
        
        # Extract common metadata for convenience
        if probes:
            self.model_name = probes[0].config.model_name
            self.hook_point = probes[0].config.hook_point
            self.hook_layer = probes[0].config.hook_layer
        
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot products with all probes.
        
        Args:
            acts: Activations to project, shape [..., d_model]
            
        Returns:
            Projected values, shape [..., n_vectors]
        """
        # Stack all vectors into a matrix
        weight_matrix = torch.stack([p.get_direction() for p in self.probes])
        
        # Project all at once
        return torch.einsum("...d,nd->...n", acts, weight_matrix)
    
    def __getitem__(self, idx) -> BaseProbe:
        """Get a probe by index."""
        return self.probes[idx]
    
    def __len__(self) -> int:
        """Get number of probes."""
        return len(self.probes)
        
    def save(self, directory: str) -> None:
        """Save all probes to a directory.
        
        Args:
            directory: Directory to save the probes
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save index file with common metadata
        index = {
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "hook_layer": self.hook_layer,
            "probes": []
        }
        
        # Save each probe
        for i, probe in enumerate(self.probes):
            filename = f"probe_{i}_{probe.name}.pt"
            filepath = os.path.join(directory, filename)
            probe.save(filepath)
            
            # Add to index
            index["probes"].append({
                "name": probe.name,
                "file": filename,
                "probe_type": probe.__class__.__name__
            })
            
        # Save index
        with open(os.path.join(directory, "index.json"), "w") as f:
            json.dump(index, f)
            
    @classmethod
    def load(cls, directory: str) -> "ProbeSet":
        """Load a ProbeSet from a directory.
        
        Args:
            directory: Directory containing the probes
            
        Returns:
            ProbeSet instance
        """
        # Load index
        with open(os.path.join(directory, "index.json")) as f:
            index = json.load(f)
            
        # Load each probe
        probes = []
        for entry in index["probes"]:
            filepath = os.path.join(directory, entry["file"])
            
            # Determine the probe class
            probe_type = entry.get("probe_type", "LinearProbe")
            if probe_type == "LinearProbe":
                from probity.probes.linear_probe import LinearProbe
                probe = LinearProbe.load(filepath)
            elif probe_type == "LogisticProbe":
                from probity.probes.linear_probe import LogisticProbe
                probe = LogisticProbe.load(filepath)
            elif probe_type == "KMeansProbe":
                from probity.probes.linear_probe import KMeansProbe
                probe = KMeansProbe.load(filepath)
            elif probe_type == "PCAProbe":
                from probity.probes.linear_probe import PCAProbe
                probe = PCAProbe.load(filepath)
            elif probe_type == "MeanDifferenceProbe":
                from probity.probes.linear_probe import MeanDifferenceProbe
                probe = MeanDifferenceProbe.load(filepath)
            else:
                # Default to base class with runtime error
                raise ValueError(f"Unknown probe type: {probe_type}")
                
            probes.append(probe)
            
        return cls(probes)


class LinearProbe(BaseProbe[LinearProbeConfig]):
    """Simple linear probe that learns one or more directions in activation space.
    
    This probe implements pure linear projection without any activation functions.
    Different loss functions can be used depending on the task:
    - MSE loss: For general regression tasks
    - Cosine loss: For learning directions that match target vectors
    - L1 loss: For robust regression with less sensitivity to outliers
    """
    
    def __init__(self, config: LinearProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        # Initialize weights 
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        if config.bias:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with standardization."""
        if self.training and self.feature_mean is None:
            # Compute statistics on first forward pass during training
            self.feature_mean = x.mean(0, keepdim=True)
            self.feature_std = x.std(0, keepdim=True) + 1e-8
            
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / self.feature_std
            
        return self.linear(x)
    
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling."""
        direction = self.linear.weight.data
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        if self.feature_std is not None and not already_unscaled:
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        if self.config.normalize_weights and not already_normalized:
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                direction = direction / (torch.norm(direction) + 1e-8)
                
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction

    def get_loss_fn(self) -> nn.Module:
        """Enhanced loss function selection with better numerical stability."""
        if self.config.loss_type == "mse":
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Center and scale targets to [-1, 1] range for better stability
                target = 2 * target - 1
                mse_loss = nn.MSELoss()(pred, target)
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return mse_loss + l2_reg
                
        elif self.config.loss_type == "hinge":
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Convert targets to {-1, 1}
                target = 2 * target - 1
                # Hinge loss
                hinge_loss = torch.mean(torch.relu(1 - pred * target))
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return hinge_loss + l2_reg
                
        elif self.config.loss_type == "cosine":
            cosine_loss = nn.CosineEmbeddingLoss()
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Convert targets to {-1, 1}
                target = 2 * target - 1
                loss = cosine_loss(pred, target, target)
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return loss + l2_reg
                
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
            
        return loss_fn
        

class LogisticProbe(BaseProbe[LogisticProbeConfig]):
    """Logistic regression probe that learns directions using cross-entropy loss."""
    
    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        # Initialize weights using sklearn-like initialization
        nn.init.zeros_(self.linear.weight)
        if config.bias:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with standardization."""
        if self.training and self.feature_mean is None:
            # Compute statistics on first forward pass during training
            self.feature_mean = x.mean(0, keepdim=True)
            self.feature_std = x.std(0, keepdim=True) + 1e-8
            
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / self.feature_std
            
        return self.linear(x)
    
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling."""
        direction = self.linear.weight.data
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        if self.feature_std is not None and not already_unscaled:
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        if self.config.normalize_weights and not already_normalized:
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                direction = direction / (torch.norm(direction) + 1e-8)
                
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction
    
    def get_loss_fn(self) -> nn.Module:
        """Get binary cross entropy loss with L2 regularization."""
        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Binary cross entropy loss
            bce_loss = nn.BCEWithLogitsLoss()(pred, target)
            # L2 regularization (matching sklearn's default C=1.0)
            l2_lambda = 0.01  # C=1.0 in sklearn corresponds to reg_lambda=0.01
            l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
            return bce_loss + l2_reg
            
        return loss_fn


class KMeansProbe(BaseProbe[KMeansProbeConfig]):
    """K-means clustering based probe that finds directions through centroids."""
    
    def __init__(self, config: KMeansProbeConfig):
        super().__init__(config)
        self.kmeans = KMeans(
            n_clusters=config.n_clusters,
            n_init=config.n_init,
            random_state=config.random_state
        )
        self.direction: Optional[torch.Tensor] = None
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit K-means and compute direction from centroids."""
        # Convert to numpy for sklearn
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Fit K-means
        self.kmeans.fit(x_np)
        centroids = self.kmeans.cluster_centers_
        
        # Determine positive and negative centroids based on cluster assignments
        labels = self.kmeans.labels_
        cluster_labels = np.zeros(self.config.n_clusters)
        for i in range(self.config.n_clusters):
            mask = labels == i
            if mask.any():
                cluster_labels[i] = np.mean(y_np[mask])
        
        pos_centroid = centroids[np.argmax(cluster_labels)]
        neg_centroid = centroids[np.argmin(cluster_labels)]
        
        # Direction is from negative to positive centroid
        direction = torch.tensor(
            pos_centroid - neg_centroid, 
            device=self.config.device,
            dtype=self.dtype
        )
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
        return self.direction


class PCAProbe(BaseProbe[PCAProbeConfig]):
    """PCA-based probe that finds directions through principal components."""
    
    def __init__(self, config: PCAProbeConfig):
        super().__init__(config)
        self.pca = PCA(n_components=config.n_components)
        self.direction: Optional[torch.Tensor] = None
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit PCA and determine direction sign based on correlation with labels."""
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Fit PCA
        self.pca.fit(x_np)
        components = self.pca.components_
        
        # Project data onto components
        projections = np.dot(x_np, components.T)
        
        # Determine sign based on correlation with labels
        correlations = np.array([np.corrcoef(proj, y_np)[0,1] for proj in projections.T])
        signs = np.sign(correlations)
        
        # Apply signs to components
        components = components * signs[:, np.newaxis]
        
        # Get primary direction (first component)
        direction = torch.tensor(
            components[0], 
            device=self.config.device,
            dtype=self.dtype
        )
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
        # Ensure consistent dtype
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
        return self.direction


class MeanDifferenceProbe(BaseProbe[MeanDiffProbeConfig]):
    """Probe that finds direction through mean difference between classes."""
    
    def __init__(self, config: MeanDiffProbeConfig):
        super().__init__(config)
        self.direction: Optional[torch.Tensor] = None
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Compute direction as difference between class means."""
        # Ensure input tensors are float32
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        
        # Calculate means for positive and negative classes
        pos_mask = y == 1
        neg_mask = y == 0
        
        pos_mean = x[pos_mask].mean(dim=0)
        neg_mean = x[neg_mask].mean(dim=0)
        
        # Direction from negative to positive
        direction = pos_mean - neg_mean
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
        return self.direction
    
# alternative implementations of logistic probe for testing
@dataclass
class LogisticProbeConfigBase(ProbeConfig):
    """Base config shared by both implementations."""
    standardize: bool = True
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1

@dataclass
class SklearnLogisticProbeConfig(LogisticProbeConfigBase):
    """Config for sklearn-based probe."""
    max_iter: int = 100
    random_state: int = 42
    

class SklearnLogisticProbe(BaseProbe[SklearnLogisticProbeConfig]):
    """Logistic regression probe using scikit-learn, matching paper implementation."""
    
    def __init__(self, config: SklearnLogisticProbeConfig):
        super().__init__(config)
        self.scaler = StandardScaler() if config.standardize else None
        self.model = LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
            fit_intercept=config.bias
        )
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the probe using sklearn's LogisticRegression."""
        # Convert to numpy
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Standardize if requested
        if self.scaler is not None:
            x_np = self.scaler.fit_transform(x_np)
            
        # Fit logistic regression
        self.model.fit(x_np, y_np)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        x_np = x.cpu().numpy()
        if self.scaler is not None:
            x_np = self.scaler.transform(x_np)
        
        # Get logits
        logits = self.model.decision_function(x_np)
        return torch.tensor(logits, device=x.device)

    def get_direction(self) -> torch.Tensor:
        """Get the learned probe direction, matching paper's implementation."""
        # Get coefficients and intercept
        coef = self.model.coef_[0]  # Shape: (input_size,)
        
        if self.scaler is not None:
            # Unscale the coefficients as done in paper
            coef = coef / self.scaler.scale_
            
        # Convert to tensor and normalize if requested
        direction = torch.tensor(coef, device=self.config.device)
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        return direction
