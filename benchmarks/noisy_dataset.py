import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
from .noise import Noise


class NoisyDataset(Dataset):
    """
    Class used to wrap a dataloader and add noise to it"s input.
    """
    def __init__(self, original_dataloader, percentage=5, noise_type="random"):
        self.original_dataloader = original_dataloader
        self.percentage = percentage
        
        # select the function to add noise
        self.noise_adder = None
        if noise_type == "random":
            self.noise_adder = Noise.add_random_perturbation
        elif noise_type == "gaussian":
            self.noise_adder = Noise.add_gaussian_noise
        elif noise_type == "salt_pepper":
            self.noise_adder = Noise.add_salt_and_pepper_noise
        else:
            warnings.warn("Warn: not valid noise type.")
            self.noise_adder = Noise.add_random_perturbation


    def add_noise(self, sample):
        noisy_sample = self.noise_adder(sample, self.percentage)
        noisy_sample = np.float32(noisy_sample)
        return torch.from_numpy(noisy_sample)

    
    def __len__(self):
        return len(self.original_dataloader.dataset)
    
    
    def __getitem__(self, index):
        # check if it is a tuple
        sample = self.original_dataloader.dataset[index]
        
        if isinstance(sample, tuple):
            original_batch, target = sample
            noisy_batch = self.add_noise(original_batch)
            return noisy_batch, target
        
        noisy_sample = self.add_noise(sample)
        
        return noisy_sample
    
    