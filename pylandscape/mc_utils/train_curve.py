import torch.nn as nn
from torch import optim
from torch.types import _device
from torch.utils.data import DataLoader
from . import CurveNet



class Interpolate:
    """
    Class to handle the interpolation and sampling of the Curve net 
    """
    def __init__(
        self,
        curve: nn.Module,
        fix_start: nn.Module,
        fix_end: nn.Module,
        criterion: nn.Module,
        learning_rate: float, 
        num_bends: int = 3,
        init_linear: bool = True,
        device: _device = "cpu"
    ) -> None:
        self.model = CurveNet(curve, fix_start, num_bends)
        # load parameters in the boundaries 
        self.model.import_base_parameters(fix_start, 0)
        self.model.import_base_parameters(fix_end, num_bends-1)
        if init_linear:
            self.model.init_linear()
            
        # training 
        # TODO: add the learning rate scheduler
        self.device = device
        self.lr = learning_rate
        self.criterion = criterion
        
    
    def train_curve(self, dataloader: DataLoader, epochs: int) -> None:
        optimizer = optim.Adam(
            filter(lambda param: param.requires_grad, self.parameters()),
            lr=self.learning_rate
        )
        self.model = self.model.to(self.device)
        for epoch in range(epochs):
            loss_hist_train = 0
            for batch, target in dataloader:
                batch, target = batch.to(self.device), target.to(self.device)
                pred = self.model(batch, None)
                loss = self.criterion(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist_train += loss.item()
            loss_hist_train /= len(dataloader)
            print(f"Epochs: {epoch}/{epochs} - Train loss: {loss_hist_train:.4f}")
                
    
    def sample_model(self, dataloader: DataLoader, t: float) -> None:
        print(f"\nTesting with t = {self.t.item()}\n")
        self.model.eval()
        loss_hist_test = 0
        for batch, target in dataloader:
            batch, target = batch.to(self.device), target.to(self.device)
            pred = self.model(batch, self.t)
            loss = self.criterion(pred, target)
            loss_hist_test += loss.item()
        loss_hist_test /= len(dataloader)
        print(f"Test loss: {loss_hist_test:.4f}")
        return loss_hist_test

