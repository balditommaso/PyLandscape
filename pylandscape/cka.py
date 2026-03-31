import numpy as np
import torch
from torch import nn, tensor
from typing import Dict, List, Optional, Tuple
from .metric import Metric

class CKA(Metric):
    """
    Minibatch CKA between all pairs of leaf-module activations across two models.

    Memory strategy
    ---------------
    * Gram vectors are moved to CPU immediately after being computed on GPU,
      so only one gram vector at a time lives on the GPU.
    * Accumulator tensors (hsic_acc, hsic_self*) are kept on CPU throughout;
      only the per-batch dot-products hit the GPU briefly.
    * torch.cuda.empty_cache() is called once per batch, not per layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_layers2: Optional[int] = None,
        across_models: bool = False,
        device: Optional[torch.device] = None,
        name: str = "CKA",
    ):
        super().__init__(name)
        self.device = torch.device("cpu") if device is None else device

        if num_layers2 is None:
            num_layers2 = num_layers
            
        self.hsic_accumulator = torch.zeros((num_layers, num_layers2), device=self.device, dtype=torch.float32)
            
        self.across_models = across_models
        if across_models:
            self.hsic_accumulator1 = torch.zeros((num_layers,), device=self.device, dtype=torch.float32)
            self.hsic_accumulator2 = torch.zeros((num_layers2,), device=self.device, dtype=torch.float32)


    def _generate_gram_matrix(self, x: tensor) -> tensor:
        """
        Generate Gram matrix and preprocess to compute unbiased HSIC.

        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.

        Args:
        x: A [num_examples, num_features] matrix.

        Returns:
        A [num_examples ** 2] vector.
        """
        n = x.shape[0]
        # Flatten spatial / channel dims and compute gram on the model's device
        x_flat = x.reshape(n, -1).to(device=self.device)
        gram = x_flat @ x_flat.t()      
        gram.fill_diagonal_(0.0)
        gram = gram.to(dtype=self.hsic_accumulator.dtype)
        
        n = gram.shape[0]

        # Row-means excluding diagonal (Szekely & Rizzo U-statistic)
        row_sum = gram.sum(dim=1)          
        means = row_sum / tensor(n - 2, device=self.device, dtype=gram.dtype)
        means = means - means.sum() / (2.0 * tensor(n - 1, device=self.device, dtype=gram.dtype))

        # Center rows and columns, then zero diagonal again
        gram = gram - means[:, None] - means[None, :]
        gram.fill_diagonal_(0.0)

        # Move to CPU immediately to free GPU memory
        return gram.reshape(-1).cpu()
    
    
    def update_state(self, activations: List[tensor]) -> None:
        layer_grams = [self._generate_gram_matrix(x).to(self.device) for x in activations]
        layer_grams = torch.stack(layer_grams, dim=0)   # stuck by the batch size
        G = layer_grams @ layer_grams.t()
        self.hsic_accumulator = self.hsic_accumulator + G
        
        
    def update_state_across_models(self, activations1: List[tensor], activations2: List[tensor]) -> None:
        n1, n2 = self.hsic_accumulator.shape
        assert n1 == len(activations1) and n2 == len(activations2), \
            f"Number of activations does not math the number of layers: {n1, n2} - {len(activations1), len(activations2)}"
            
        layer_grams1 = [self._generate_gram_matrix(x).to(self.device) for x in activations1]
        layer_grams1 = torch.stack(layer_grams1, dim=0)
        layer_grams2 = [self._generate_gram_matrix(x).to(self.device) for x in activations2]
        layer_grams2 = torch.stack(layer_grams2, dim=0)
        
        G = layer_grams1 @ layer_grams2.t()
        self.hsic_accumulator = self.hsic_accumulator + G

        G1 = torch.einsum("ij,ij->i", layer_grams1, layer_grams1)
        self.hsic_accumulator1 = self.hsic_accumulator1 + G1
        
        G2 = torch.einsum("ij,ij->i", layer_grams2, layer_grams2)
        self.hsic_accumulator2 = self.hsic_accumulator2 + G2
        
        
    def result(self) -> tensor:
        mean_hsic = self.hsic_accumulator.clone()
        
        if self.across_models:
            norm1 = torch.sqrt(self.hsic_accumulator1)
            norm2 = torch.sqrt(self.hsic_accumulator2)
            
            mean_hsic = mean_hsic / norm1[:, None]
            mean_hsic = mean_hsic / norm2[None, :]
        else:
            norm = torch.sqrt(torch.diag(mean_hsic))
            mean_hsic = mean_hsic / norm[:, None]
            mean_hsic = mean_hsic / norm[None, :]
        
        self.results["CKA_heatmap"] = mean_hsic
        
        return mean_hsic
        
        
