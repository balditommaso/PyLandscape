import numpy as np
import pandas as pd
import torch
from typing import Union



class Noise:
    """
    Class with static methods to add different kind of noise
    to the input data
    """
    
    @staticmethod
    def set_seed(seed: int):
        """Set a seed of the randomness in the noise computations.

        Args:
            seed (int): key of the seed.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    

    @staticmethod
    def add_random_perturbation(
        data: Union[torch.tensor, np.ndarray], 
        percentage: float = 5.0, 
        perturbation_range: float = 0.1
    ) -> np.ndarray:
        """Add random perturbation on a tensor of data.

        Args:
            data (Union[torch.tensor, np.ndarray]): Data tensor to be perturbed.
            percentage (float, optional): Percentage of noise to be added. Defaults to 5.0.
            perturbation_range (float, optional): Max possible perturbation. Defaults to 0.1.

        Returns:
            np.ndarray: Perturbed version of the input tensor.
        """
        if isinstance(data, torch.tensor):
            data = data.clone()
        else:
            data = data.copy()
        
        perturbation = np.random.uniform(0, perturbation_range, data.shape)
        noisy_data = data + (percentage / 100) * perturbation
        return noisy_data
    
    
    @staticmethod
    def add_gaussian_noise(
        data: Union[torch.tensor, np.ndarray], 
        percentage: float = 5.0, 
        mean: float = 0.0, 
        std: float = 1
    ) -> np.ndarray:
        """Add Gaussian perturbation on a tensor of data.

        Args:
            data (Union[torch.tensor, np.ndarray]): Data tensor to be perturbed.
            percentage (float, optional): Percentage of noise to be added. Defaults to 5.0.
            mean (float, optional): Mean of the Gaussian distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the Gaussian distribution. Defaults to 1.

        Returns:
            np.ndarray: Perturbed version of the input tensor.
        """
        if isinstance(data, torch.Tensor):
            data = data.clone()
        else:
            data = data.copy()
        noise = np.random.normal(mean, std, data.shape)
        noisy_data = data + (percentage / 100) * noise
        return noisy_data
    
    
    @staticmethod
    def add_salt_and_pepper_noise(
        data: Union[torch.tensor, np.ndarray, pd.DataFrame], 
        percentage: float = 5.0
    ) -> Union[torch.tensor, np.ndarray, pd.DataFrame]:
        """Add Salt & Pepper perturbation on a tensor of data.

        Args:
            data (Union[torch.tensor, np.ndarray, pd.DataFrame]): Data tensor to be perturbed.
            percentage (float, optional): Percentage of noise to be added. Defaults to 5.0.

        Raises:
            ValueError: Not supported input type.

        Returns:
            Union[torch.tensor, np.ndarray, pd.DataFrame]: Perturbed version of the input tensor.
        """
        data_type = type(data)
        if isinstance(data, torch.Tensor):
            data = data.clone().cpu().numpy()
        elif isinstance(data, pd.DataFrame):
            data = data.copy().values
        elif isinstance(data, np.ndarray):
            data = data.copy()
        else:
            raise ValueError(
                "Unsupported data type. Only PyTorch tensors, Pandas DataFrames, and NumPy ndarrays are supported."
            )
        
        # Calculate the number of elements to corrupt
        num_elements = data.size
        num_corrupted = int(num_elements * (percentage / 200))
        
        # Add salt noise
        coords = np.unravel_index(
            np.random.choice(num_elements, num_corrupted, replace=False), data.shape
        )
        data[coords] = 1
        
        # Add pepper noise
        coords = np.unravel_index(
            np.random.choice(num_elements, num_corrupted, replace=False), data.shape
        )
        data[coords] = 0
        
        if data_type == np.ndarray:
            return data
        elif data_type == pd.DataFrame:
            return pd.DataFrame(data)
        elif data_type == torch.Tensor:
            return torch.from_numpy(data)
    
    