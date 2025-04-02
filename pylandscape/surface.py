import torch
import numpy as np
import pyhessian
from torch import nn, tensor
from copy import deepcopy
from torch.types import _device
from torch.utils.data import DataLoader
from .metric import Metric
from collections import OrderedDict 
from typing import Optional, List, Dict, Tuple, Sequence



class Surface(Metric):
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: _device = "cpu", 
        seed: Optional[int] = 20000605, 
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
    def get_params(model: nn.Module, directions: List[tensor], step_values: List[float]) -> nn.Module:
        """
        Generate a new model by perturbing its parameters along multiple directions.

        Args:
            model (nn.Module): Original target model.
            directions (List[torch.Tensor]): List of N direction tensors for perturbation.
            step_values (List[float]): List of N magnitudes for perturbations.

        Returns:
            nn.Module: Model shifted in the loss landscape.
        """
        assert len(directions) == len(step_values), "Number of directions and step values must match!"
        
        perturbed_model = deepcopy(model)
        
        # iterate over model parameters and apply perturbations in that direction
        for (name, module), perturbed_module, *dir_vect in zip(
            model.named_parameters(), perturbed_model.parameters(), *directions
        ):
            assert all(d.shape == module.data.shape for d in dir_vect), \
                f"Tensor mismatch while adding perturbation! ({name})"
                
            # apply perturbation
            perturbation = sum(step * d for step, d in zip(step_values, dir_vect))
            perturbed_module.data = module.data + perturbation

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
    
    
    
    def _compute_hyperplane(self, v: List[List[tensor]], steps: List[float]) -> np.ndarray:
        N = len(v)
        shape = (len(steps),) * N
        loss_surface = np.zeros(shape)
        for indices in np.ndindex(shape):
            step_values = [steps[idx] for idx in indices]
            perturbed_model = self.get_params(self.model, v, step_values)
            # Compute and store loss
            loss_surface[indices] = self.criterion(perturbed_model(self.inputs), self.targets).item()

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
        loss_list = self._compute_hyperplane([v], lams)  
            
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
        
        loss_surface = self._compute_hyperplane([v1, v2], lams)
        
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
        
        loss_list = self._compute_hyperplane(top_eigenvector, lams)
        
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
        
        loss_surface = self._compute_hyperplane(top_eigenvector, lams)
        
        self.results["hessian_plane"] = {"alpha": lams, "beta": lams, "loss": loss_surface}
        return lams, lams, loss_surface
    
    
    def hessian_hyperplane(
        self,
        lams: Tuple[float, float],
        steps: int,
        n: int,
        max_iter: int = 100
    ) -> Sequence[np.ndarray]:
        # get the top n eigenvectors
        hessian_comp = pyhessian.hessian(self.model, 
                                         self.criterion, 
                                         dataloader=self.dataloader, 
                                         cuda=self.device.type == "cuda")
        _, top_eigenvector = hessian_comp.eigenvalues(maxIter=max_iter, tol=1e-5, top_n=n)
        
        # coefficients to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        loss_hyperplane = self._compute_hyperplane(top_eigenvector, lams)
        return lams, loss_hyperplane
    

    def random_hyperplane(
        self,
        lams: Tuple[float, float],
        steps: int,
        n: int
    ) -> Sequence[np.ndarray]:
        
        # generate the first random vector
        v1 = Surface._rand_like(self.model.parameters())
        v1 = pyhessian.utils.normalization(v1)
        directions = [v1]
        for _ in range(n-1):
            v = self._rand_like(v1)
            v = self.orthogonalize_vectors(v, directions[-1])
            v = pyhessian.utils.normalization(v)
            directions.append(v)
                
        # coefficients to perturb the model
        min_lam, max_lam = lams
        lams = np.linspace(min_lam, max_lam, steps).astype(np.float32)
        
        loss_hyperplane = self._compute_hyperplane(directions, lams)
        return lams, loss_hyperplane
        