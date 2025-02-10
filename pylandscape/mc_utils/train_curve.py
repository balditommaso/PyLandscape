import torch
import torch.nn as nn
from tqdm import tqdm
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
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=self.lr
        )
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            loss_hist_train = 0
            
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                
                for batch, target in tepoch:
                    batch, target = batch.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = self.model(batch, None)
                    loss = self.criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    loss_hist_train += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    
                loss_hist_train /= len(dataloader)
                
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {loss_hist_train:.4f}")
                
    
    def sample_model(self, dataloader: DataLoader, t: float) -> None:
        self.model.eval()
        loss_hist_test = 0
        with torch.no_grad():
            with tqdm(dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Testing with t = {t.item():.4f}")
                
                for batch, target in tepoch:
                    batch, target = batch.to(self.device), target.to(self.device)
                    
                    pred = self.model(batch, t)
                    loss = self.criterion(pred, target)
                    
                    loss_hist_test += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                
        loss_hist_test /= len(dataloader)
        print(f"\n###\tTest loss: {loss_hist_test:.4f}\t###\n")
        return loss_hist_test

