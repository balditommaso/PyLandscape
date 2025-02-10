
"""
The following code is adapted from:
DO WIDE AND DEEP NETWORKS LEARN THE SAME
THINGS? UNCOVERING HOW NEURAL NETWORK
REPRESENTATIONS VARY WITH WIDTH AND DEPTH
Thao Nguyen, AI Resident, Google Research
https://blog.research.google/2021/05/do-wide-and-deep-networks-learn-same.html
"""

import numpy as np
import torch
from torch import tensor, nn
from typing import Dict, Any
from torch.types import _device
from torch.utils.data import DataLoader
from .metric import Metric


class CKA(Metric):
    
    def __init__(
        self, 
        device: _device = 'cpu', 
        name: str = "CKA_similarity", 
    ) -> None:
        super().__init__(name)
        self.device = device

    
    @staticmethod
    def gram_matrix(X: tensor):
        '''
        Generate Gram matrix and preprocess to compute unbiased HSIC.

        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.

        Args:
        x: A [num_examples, num_features] matrix.

        Returns:
        A [num_examples ** 2] vector.
        '''
        X = X.reshape(X.shape[0], -1)
        gram_X = X @ X.T
        
        if not torch.allclose(gram_X, gram_X.t()):
            raise ValueError("Gram matrix should be symmetric!")
        
        means = torch.mean(gram_X, 0)
        means -= torch.mean(means) / 2
        gram_X -= means[:, None]
        gram_X -= means[None, :]
        
        return gram_X.reshape((-1,))
    
    
    @staticmethod
    def update_state(hsic_accumulator: tensor, activations: Dict[str, tensor]) -> tensor:
        layers_gram = []
        for x in activations.values():
            if x is None:
                continue
            layers_gram.append(CKA.gram_matrix(x))
        layers_gram = torch.stack(layers_gram, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram, layers_gram.T)
    
    
    @staticmethod
    def update_state_across_models(
        hsic_accumulator: tensor,
        hsic_accumulator1: tensor, 
        activations1: Dict[str, Any], 
        hsic_accumulator2: tensor, 
        activations2: Dict[str, Any]
    ) -> tensor:
        # dimension test
        torch.testing.assert_close(hsic_accumulator1.shape[0], len(activations1))
        torch.testing.assert_close(hsic_accumulator2.shape[0], len(activations2))
        # activation 1
        layers_gram1 = []
        for x in activations1.values():
            if x is None:
                continue
            # HAWQ nesting problem
            elif isinstance(x, tuple):
                layers_gram1.append(CKA.gram_matrix(x[0]))    
            else:
                layers_gram1.append(CKA.gram_matrix(x))
        layers_gram1 = torch.stack(layers_gram1, axis=0)
        # activation 2
        layers_gram2 = []
        for x in activations2.values():
            if x is None:
                continue
            # HAWQ nesting problem
            elif isinstance(x, tuple):
                layers_gram2.append(CKA.gram_matrix(x[0]))    
            else:
                layers_gram2.append(CKA.gram_matrix(x))
        layers_gram2 = torch.stack(layers_gram2, axis=0)
        return hsic_accumulator + torch.matmul(layers_gram1, layers_gram2.T), \
                hsic_accumulator1 + torch.einsum('ij,ij->i', layers_gram1, layers_gram1), \
                hsic_accumulator2 + torch.einsum('ij,ij->i', layers_gram2, layers_gram2)
    
    
    def output_similarity(
        self, 
        model1: nn.Module,
        model2: nn.Module, 
        dataloader: DataLoader,
        num_outputs: int = 10, 
        num_runs: int = 1,
        aggregate: str = "mean",
    ) -> float:
        """_summary_

        Args:
            model1 (nn.Module): First model to be compared.
            model2 (nn.Module): Second model to be compared.
            dataloader (DataLoader): Dataloader used to get the input data.
            num_outputs (int, optional): Number of output batches to be concatenated. Defaults to 10.
            num_runs (int, optional): How many runs should we repeat the computation. Defaults to 1.
            aggregate (str, optional): NumPy operator to use to aggregate the results 
                of different runs. Defaults to "mean".

        Returns:
            float: CKA similarity between two models' output
        """
        model1.eval()
        model2.eval()
        cka_similarity = []
        for _ in range(num_runs):
            # output of models
            F1 = []
            F2 = []
            for i, (batch, _) in enumerate(dataloader, 1):
                batch = batch.to(self.device)
                F1.append(model1(batch))
                F2.append(model2(batch))
                # stop condition
                if i > num_outputs:
                    break
            
            F1 = torch.cat(F1)
            F2 = torch.cat(F2)
            
            gram_x = CKA.gram_matrix(F1)
            gram_y = CKA.gram_matrix(F2)
            
            scaled_hsic = torch.ravel(gram_x) @ torch.ravel(gram_y)
            normalization_x = torch.linalg.norm(gram_x)
            normalization_y = torch.linalg.norm(gram_y)
            s = scaled_hsic / (normalization_x * normalization_y)
            if torch.isnan(s):
                cka_similarity.append(0.0)
            else:
                cka_similarity.append(s.item())
        
        # aggregate the result
        result = getattr(np, aggregate)(cka_similarity)

        return result
    
    
    # TODO: compute similarity using model parameters
    def module_similarity(self):
        pass

