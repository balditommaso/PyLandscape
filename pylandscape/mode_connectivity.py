import torch
from torch.types import _device
from torch import nn, tensor
from torch.utils.data import DataLoader
from typing import Callable
from . import Metric
from .mc_utils import curves, Interpolate

class ModeConnectivity(Metric):
    def __init__(self, device: torch.device = "cpu", name: str = "mode_connectivity") -> None:
        super().__init__(name)
        self.device = device
        
    
    @staticmethod
    def max_deviation(loss: tensor) -> float:
        '''
        Compute the max deviation from the average of the boundaries.
        '''
        # calculate the midpoint of the first and last elements
        midpoint = (loss[0] + loss[-1]) / 2

        # calculate the absolute deviation from the midpoint for each element in loss
        deviations = torch.abs(loss - midpoint)

        # find the index with the maximum deviation
        max_dev_idx = torch.argmax(deviations)

        return (midpoint - loss[max_dev_idx]).item()

        
    def compute(
        self, 
        model1: nn.Module,
        model2: nn.Module,
        criterion: Callable,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        learning_rate: float,
        curve: str = "Bezier", 
        num_bends: int = 3, 
        num_points: int = 30,
        max_epochs: int = 50,
        init_linear: bool = True,
        device: _device = "cpu"
    ) -> float:
        # select the curve for the interpolation
        curve = getattr(curves, curve)
        

        # interpolate the functions
        interpolate = Interpolate(
            curve, 
            fix_start=model1, 
            fix_end=model2,
            criterion=criterion,
            learning_rate=learning_rate,
            num_bends=num_bends,
            init_linear=init_linear,
            device=device
        )
        
        # train and shape the bezier curve
        interpolate.train_curve(train_dataloader, max_epochs)
        
        # evaluate the model for each t
        ts = torch.linspace(0.0, 1.0, num_points)
        loss = torch.zeros(num_points)
        
        t = torch.FloatTensor([0.0]).to(self.device)
        for i, t_value in enumerate(ts):
            t.data.fill_(t_value)
            loss[i] = interpolate.sample_model(test_dataloader, t)
        # return the max deviation
        return self.max_deviation(loss)