import torch
from torch import nn, tensor
from copy import deepcopy
from torch.types import _device
from torch.utils.data import DataLoader
from .metric import Metric
from collections import OrderedDict 
from typing import Optional, List, Dict, Tuple
import numpy as np
import pyhessian


class Surface(Metric):
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: _device = "cpu", 
        seed: Optional[int] = None, 
        name: str = "plot"
    ) -> None:
        super().__init__(name)
        self.model = model
        self.criterion = criterion
        self.device = device
        self.seed = seed     
        # preparing the input for loss evaluation
        self.dataloader = dataloader
        inputs, targets = iter(dataloader).__next__()
        self.inputs, self.targets = inputs.to(self.device), targets.to(self.device)
        
        
    @staticmethod
    def named_eigenvectors(module: nn.Module, eigenvectors: List[tensor]) -> Dict[str, tensor]:
        named_eigenvectors = OrderedDict()
        
        for (name, param), v in zip(module.named_parameters(), eigenvectors):
            if param.shape == v.shape:
                named_eigenvectors[name] = v
            else:
                print(f"Warning: shape miss match with ({name})!")
                
        return named_eigenvectors
    
    
    @staticmethod
    def get_params(
        model: nn.Module, 
        direction1: tensor, 
        alpha: float,
        direction2: Optional[tensor] = None,
        beta: Optional[float] = None
    ) -> nn.Module:
        """
        Generate a new model perturbing the parameters in the specified directions with a 
        certain magnitude.

        Args:
            model (nn.Module): Original target model.
            direction1 (tensor): First direction of the perturbation.
            alpha (float): Magnitude of the perturbation on direction1. 
            direction2 (Optional[tensor], optional): Second direction of the perturbation. Defaults to None.
            beta (Optional[float], optional): Magnitude of the perturbation on direction2. Defaults to None.

        Returns:
            nn.Module: model shifted in the loss landscape.
        """
        perturbed_model = deepcopy(model)
        
        if direction2 is None or beta is None:
            # single line (2D)
            for (name, module), perturbed_module, d in zip(model.named_parameters(), perturbed_model.parameters(), direction1):
                assert d.shape == module.data.shape and module.data.shape == perturbed_module.data.shape, \
                    f"Tensor mismatch while adding perturbation! ({name})"
                        
                perturbed_module.data = module.data + alpha * d
        else:
            # surface (3D)
            for (name, module), perturbed_module, d1, d2 in zip(model.named_parameters(), perturbed_model.parameters(), direction1, direction2):
                assert d1.shape == module.data.shape and module.data.shape == perturbed_module.data.shape and \
                       d2.shape == module.data.shape and module.data.shape == perturbed_module.data.shape, \
                    f"Tensor mismatch while adding perturbation! ({name})"
                
                perturbed_module.data = module.data + alpha * d1 + beta * d2
                    
        return perturbed_model
    
    
    @staticmethod
    def _rand_like(vector: List[tensor]) -> List[tensor]:
        """
        Similar to `torch.rand_like` but for a list of tensors.

        Args:
            vector (List[tensor]): List of tensors with different shapes

        Returns:
            List[tensor]: List of tensors with random values with the same shape 
                          of "vector".
        """
        return [torch.rand_like(v) for v in vector]
    
    
    @staticmethod
    def orthogonalize_vectors(new_vector: List[tensor], vector: List[tensor]) -> List[tensor]:
        """
        Orthogonalizes new_vector with respect to vector. Both are lists of torch tensors.
        
        Args:
        new_vector (list of tensor): The list of tensors to be orthogonalized.
        vector (list of tensor): The list of reference tensors (to orthogonalize against).
        
        Returns:
        list of tensor: Orthogonalized version of new_vector.
        """
        orthogonal_vector = []
        
        # iterate over corresponding tensors in new_vector and vector
        for new_tensor, ref_tensor in zip(new_vector, vector):
            # compute dot product of new_tensor and ref_tensor
            dot_product = torch.dot(new_tensor.flatten(), ref_tensor.flatten())
            
            # compute the squared norm of the ref_tensor
            ref_norm_sq = torch.norm(ref_tensor, p=2).pow(2)
            
            # orthogonalize new_tensor with respect to ref_tensor
            orthogonalized_tensor = new_tensor - (dot_product / ref_norm_sq) * ref_tensor
            
            # append the orthogonalized tensor to the result list
            orthogonal_vector.append(orthogonalized_tensor)
        
        return orthogonal_vector
    
    
    @staticmethod
    def check_orthogonality(v1: List[tensor], v2: List[tensor], tol: float = 1e-6) -> bool:
        """
        Check the orthogonality between the new_vector and vector by calculating the dot product.
        
        Args:
        new_vector (list of tensor): Orthogonalized vector.
        vector (list of tensor): Reference vector.
        tolerance (float): Numerical tolerance for orthogonality check. Default is 1e-6.
        
        Returns:
        bool: True if orthogonal within tolerance, False otherwise.
        """
        for t1, t2 in zip(v1, v2):
            # Compute the dot product between the corresponding tensors
            dot_product = torch.dot(t1.flatten(), t2.flatten())            
            # Check if the dot product is close to 0 within the tolerance
            if torch.abs(dot_product) > tol:
                return False
        return True
        
    
    def _compute_line(self, v: List[tensor], steps: List[float]) -> List[float]:
        losses = []
        for step in steps:
            perturbed_model = self.get_params(self.model, v, step).to(self.device)
            losses.append(self.criterion(perturbed_model(self.inputs), self.targets).item())
        
        return losses
    
    
    def _compute_surface(self, v1: List[tensor], v2: List[tensor], steps: List[float]) -> List[float]:
        loss_surface = np.zeros((len(steps), len(steps)))
        # compute the loss in the mesh of points
        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps):
                perturbed_model = self.get_params(self.model, v1, step1, v2, step2)
                loss_surface[i, j] = self.criterion(perturbed_model(self.inputs), self.targets).item()
                
        return loss_surface
    
    
    def random_line(self, lams: Tuple[float, float], steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the loss plot of model perturbed on a random direction. 

        Args:
            lams (Tuple[float, float]): Tuple with minimum and maximum perturbations.
            steps (int): Number of steps between the minimum and maximum perturbations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of NumPy arrays with the points along random direction
                                           and the loss computed in these points.
        """
        # generate a random vector and normalize it
        v = Surface._rand_like(self.model.parameters())
        v = pyhessian.utils.normalization(v)
        
        # coefficient to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        # compute the loss along the line        
        loss_list = self._compute_line(v, lams)
            
        self.results["random_line"] = {"alpha": lams, "loss": loss_list}
        return lams, loss_list
    
    
    def random_surface(self, lams: Tuple[float, float], steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the loss plot of model perturbed on a random direction and its orthogonal.

       Args:
            lams (Tuple[float, float]): Tuple with minimum and maximum perturbations.
            steps (int): Number of steps between the minimum and maximum perturbations.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of NumPy arrays with the points along 
                                                       random direction and the loss computed in these points.
        """
        # generate the first random vector
        v1 = Surface._rand_like(self.model.parameters())
        v1 = pyhessian.utils.normalization(v1)
        
        # generate the second random vector, orthogonal to the first
        v2 = self._rand_like(v1)
        v2 = self.orthogonalize_vectors(v2, v1)
        v2 = pyhessian.utils.normalization(v2)
        
        assert self.check_orthogonality(v1, v2), "The two vectors are not orthogonal!"
        
        # coefficients to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        loss_surface = self._compute_surface(v1, v2, lams)
        
        self.results["random_plane"] = {"alpha": lams, "beta": lams, "loss": loss_surface}
        return lams, lams, loss_surface
    
    
    def hessian_line(
        self, 
        lams: Tuple[float, float], 
        steps: int, 
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the loss plot of model perturbed along the top eigenvector of the model. 

        Args:
            lams (Tuple[float, float]): Tuple with minimum and maximum perturbations.
            steps (int): Number of steps between the minimum and maximum perturbations.
            max_iter (int): Max Number of iteration to compute the eigenvectors. Default is 100.
        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of NumPy arrays with the points along 
                                           top eigenvector direction and the loss computed 
                                           in these points.
        """
        # get the top eigenvectors as direction
        hessian_comp = pyhessian.hessian(self.model, 
                                         self.criterion, 
                                         dataloader=self.dataloader, 
                                         cuda=self.device.type == "cuda")
        _, top_eigenvector = hessian_comp.eigenvalues(maxIter=max_iter, tol=1e-5)
        # coefficient to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        loss_list = self._compute_line(top_eigenvector[0], lams)
        
        self.results["hessian_line"] = {"alpha": lams, "loss": loss_list}
        return lams, loss_list
    
    
    def hessian_surface(
        self, 
        lams: Tuple[float, float], 
        steps: int, 
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the loss plot of model perturbed along the top-2 eigenvectors of the model. 

        Args:
            lams (Tuple[float, float]): Tuple with minimum and maximum perturbations.
            steps (int): Number of steps between the minimum and maximum perturbations.
            max_iter (int): Max Number of iteration to compute the eigenvectors. Default is 100.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple of NumPy arrays with the points along 
                                                       top-2 eigenvector directions and the loss computed 
                                                       in these points.
        """
        # get the top eigenvectors as direction
        hessian_comp = pyhessian.hessian(self.model, 
                                         self.criterion, 
                                         dataloader=self.dataloader, 
                                         cuda=self.device.type == "cuda")
        _, top_eigenvector = hessian_comp.eigenvalues(maxIter=max_iter, tol=1e-5, top_n=2)
        # coefficients to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        loss_surface = self._compute_surface(top_eigenvector[0], top_eigenvector[1], lams)        
        
        self.results["hessian_plane"] = {"alpha": lams, "beta": lams, "loss": loss_surface}
        return lams, lams, loss_surface

    

        
