"""
Portion of code relative to FKeras inspired by: 
https://github.com/KastnerRG/fkeras
"""

from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
from pyhessian import hessian
from .metric import Metric
import torch
from torch import nn, tensor
import numpy as np

# NOTE: there is a bug in the PyHessian package, when it get the parameters w/o grads


METRICS = ["trace", "eigenvalue", "eigenvector", "density", "fkeras"]

class Hessian(Metric):
    """
    Wrapper of the PyHessian package: https://github.com/amirgholami/PyHessian
    """
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module,
        dataloader: DataLoader, 
        name: str = "hessian",
        use_cache: bool = True,
    ) -> None:
        """
        Initializer of the Hessian class.

        Args:
            model (nn.Module): target model.
            criterion (nn.Module): loss criterion to compute the gradients.
            dataloader (DataLoader): Dataloader to retrieve the input data.
            name (str, optional): Name assigned to the pickle file. Defaults to "hessian".
            use_cache (bool, optional): Flag to activate the possibility to re-use 
                                        already computed metrics. Defaults to True.
        """
        super().__init__(name)
        self.model = model
        self.use_cache = use_cache
        # init the PyHessian Class
        self.hessian_comp = hessian(
            self.model,
            criterion=criterion,
            dataloader=dataloader,
            cuda=torch.cuda.is_available()
        )
        
        
    def are_top_n_available(self, n: int = 1) -> bool:
        """
        Check it the top-n eigenvectors and eigenvalues are available.

        Args:
            n (int, optional): Number of top eigenvectors and eigenvalues. Defaults to 1.

        Returns:
            bool: True if they are available or false otherwise.
        """
        if hasattr(self.results, "eigenvalue") and hasattr(self.results, "eigenvector"):
            if len(self.results["eigenvalue"]) >= n and len(self.results["eigenvector"]) >= n:
                return True
        return False
            
            
    def sum_hessian_rank(
        self, 
        params: List[tensor], 
        eigenvectors: List[tensor], 
        eigenvalues: Optional[List[tensor]] = None
    ) -> tensor:
        """
        Given flattened list of parameters, list of eigenvectors, and list of
        eigenvalues, compute the eigenvector/value scores.

        Combine the weight eigenvectors into a single vector for model-wide
        parameter sensitivity ranking using weighted sum strategy.
        Return a list of eigenvector/eigenvalue
        scores, one score for each parameter.
        Current method: weighted sum of eigenvectors
        """
        flattened_params = torch.cat([param.flatten() for param in params], dim=0)
        combined_eigenvector_score = torch.zeros_like(flattened_params)
        for i in range(len(eigenvectors)):
            combined_eigenvector = []
            curr = 0
            for j in range(0, len(eigenvectors[i])):
                if params[curr].numel() != eigenvectors[i][j].numel():
                    # skip what is not a weight
                    continue
                combined_eigenvector.append(eigenvectors[i][j].flatten())
                curr += 1
                if curr >= len(params):
                    # you have found all the matches
                    break
            combined_eigenvector = torch.cat(combined_eigenvector, dim=0)
            scalar_rank = torch.dot(combined_eigenvector, flattened_params)
            if eigenvalues:
                scalar_rank *= eigenvalues[i]
            combined_eigenvector_score += torch.abs(scalar_rank * combined_eigenvector)
        return combined_eigenvector_score
        
        
    def hessian_rank(
        self, 
        n_iter: int = 100,
        tol: float = 1e-6,
        top_n: int = 1
    ) -> Tuple[tensor, tensor]:
        """
        Given list of eigenvectors and eigenvalues, compute the sensitivity.
        Use Hessian to rank parameters based on sensitivity to bit flips with
        respect to all parameters (not by layer like layer_hessian_ranking).

        Args:
            n_iter (int, optional): Maximum number of iteration of the power methods. Defaults to 100.
            tol (float, optional): Tolerance of difference between two iterations. Defaults to 1e-6.
            top_n (int, optional): Number of top eigenvalues and eigenvectors to use. Defaults to 1.

        Returns:
            Tuple[tensor, tensor]: Tuple with respectively the parameters ranking and the score of each parameter
        """
        # get the list of weights 
        params = []
        for _, layer in self.model.named_modules():
            if hasattr(layer, 'quant_weight'): # selecting only the quantized module
                params.append(layer.weight)
        # cumpute the Hessian metrics or used already stored
        if self.use_cache and self.are_top_n_available(n=top_n):
            eigenvalues, eigenvectors = self.results["eigenvalue"][:top_n], self.results["eigenvector"][:top_n]
        else:
            eigenvalues, eigenvectors = self.hessian_comp.eigenvalues(n_iter, tol, top_n)
            self.results["eigenvalue"] = eigenvalues
            self.results["eigenvector"] = eigenvectors
            
        eigenvectors_rank = self.sum_hessian_rank(params, eigenvectors, eigenvalues)

        param_ranking = torch.argsort(torch.abs(eigenvectors_rank), descending=True)
        param_scores = eigenvectors_rank[param_ranking]
        self.results["param_ranking"] = param_ranking
        self.results["param_score"] = param_scores
        return param_ranking, param_scores


    def compute_trace(self, n_iter: int = 100, tol: float = 1e-6, aggregate: str = "sum") -> np.ndarray:  
        """
        Compute the trace of the Hessian matrix of the model with PyHessian package.

        Args:
            n_iter (int, optional): Maximum number of iteration of the power methods. Defaults to 100.
            tol (float, optional): Tolerance of difference between two iterations. Defaults to 1e-6.
            aggregate (str, optional): NumPy operator used to aggregate the trace. Defaults to "sum".

        Returns:
            np.array: Aggregate value of the trace
        """
        trace = self.hessian_comp.trace(maxIter=n_iter, tol=tol)
        self.results["trace"] = getattr(np, aggregate)(trace)
        return trace
    
    
    def compute_eigenvalues(
        self, 
        n_iter: int = 100, 
        tol: float = 1e-6, 
        top_n: int = 1, 
    ) -> Tuple[List[tensor], List[tensor]]:
        """
        Compute the top eigenvalues and eigenvector of the Hessian matrix of the model
        with PyHessian

        Args:
            n_iter (int, optional): Maximum number of iteration of the power methods. Defaults to 100.
            tol (float, optional): Tolerance of difference between two iterations. Defaults to 1e-6.
            top_n (int, optional): Number of top eigenvalues and eigenvectors to use. Defaults to 1.
            
        Returns:
            Tuple[List[tensor], List[tensor]]: Tuple with the list of eigenvalues and eigenvectors.
        """
        eigenvalue, eigenvector = self.hessian_comp.eigenvalues(n_iter, tol, top_n)
        self.results["eigenvalue"] = eigenvalue
        self.results["eigenvector"] = eigenvector
        return eigenvalue, eigenvector
               
  