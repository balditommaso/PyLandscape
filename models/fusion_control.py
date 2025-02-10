import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import nn, tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from collections import OrderedDict
from typing import Union, List, Tuple
from .utils import yaml_load
from benchmarks.jacobian import JacobianReg as jReg
from benchmarks.lipschitz import lipschitz_regularizer as lipReg
from datamodule import FusionData, Fusion_label_mean, Fusion_label_std
import models.quantization as quant



def fusion_model(filters_conv: List[int], neurons_dense: List[int]) -> nn.Module:
    layers_list = []
    in_channels = 1
    for i, filter in enumerate(filters_conv):
        layers_list.extend([
            (f"conv_{i}", nn.Conv2d(in_channels, filter, kernel_size=3)),
            (f"act_{i}", nn.ReLU()),
            (f"max_pool_{i}", nn.MaxPool2d(2))
        ])
        in_channels = filter
    # flatten layer
    layers_list.append(("flatten", nn.Flatten()))
    
    # Compute the in_features dynamically
    dummy_input = torch.zeros((1, 1, 32, 32))   # expected input shape
    temp_model = nn.Sequential(OrderedDict(layers_list))
    flattened_size = temp_model(dummy_input).shape[1]
    in_features = flattened_size  
    
    for i, features in enumerate(neurons_dense, len(filters_conv)):
        layers_list.extend([
            (f"dense_{i}", nn.Linear(in_features, features)),
            (f"act_{i}", nn.ReLU()),
        ])
        in_features = features
        
    # classifier
    layers_list.append(("classifier", nn.Linear(features, 2)))
    
    return nn.Sequential(OrderedDict(layers_list))
    


class FusionControl(pl.LightningModule):
    def __init__(self, 
            config: Union[str, dict],
            quantized: bool, 
            learning_rate: float,
            bit_width: int=32,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # load the config from yaml file
        if isinstance(config, str):
            config = yaml_load(config)
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.quantized = quantized
        self.bit_width = bit_width
        
        # define the model
        list_filters_conv = [16, 16, 24]
        list_neurons_dense = [42, 64]
        
        self.model = fusion_model(list_filters_conv, list_neurons_dense)
        if self.quantized:
            quant_cfg = config['model']['quantization']
            quant_cfg = dict(
                first_layer_bit_width=quant_cfg.get('first_layer_bit_width', None),
                act_bit_width=quant_cfg.get('act_bit_width', None),
                input_quant=getattr(quant, quant_cfg['input_quant']),
                verbose=quant_cfg.get('verbose', False),
            )
            self.model = quant.quantize_model(self.model, bit_width, **quant_cfg)
            
        self.criterion = nn.MSELoss()
        self.test_criterion = nn.L1Loss()
        
        self.scheduler = config['fit']['lr_scheduler']
        # regularizer
        self.l1 = config['fit']['regularizer'].get('l1', 0.0)
        self.l2 = config['fit']['regularizer'].get('l2', 0.0)
        self.jacobian = config['fit']['regularizer'].get('jacobian', 0.0)
        self.parseval = config['fit']['regularizer'].get('parseval', 0.0)
        
        self.jReg = jReg(n=1)
        self.lipReg = lipReg
        self.l1Reg = nn.L1Loss()
        
        # for testing 
        self.ampl_pred = []
        self.phase_pred = []
        self.ampl_target = []
        self.phase_target = []


    def forward(self, x: tensor):
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), 
                         lr=self.learning_rate, 
                         weight_decay=self.l2)  # lr=1e-3
        scheduler = None
        if self.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
        else:
            return optimizer
        
        print(f"Learning rate scheduler adopted: {self.scheduler}\n")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "name": "learning_rate",
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True
            }
        } 
            
            
    def training_step(self, batch: Tuple[tensor, tensor], batch_idx: int):
        input, target = batch
        input.requires_grad = True # this is essential!
        input_hat = self(input)
        loss = self.criterion(input_hat, target)
        # Jacobian regularizer
        if self.jacobian > 0.0:
            j_loss = self.jReg(input, input_hat)
            loss = loss + self.jacobian * j_loss
            self.log("j_loss", j_loss, on_epoch=True)
        # Orthogonality regularizer
        if self.parseval > 0.0:
            lip_loss = self.lipReg(self.model)
            loss = loss + self.parseval * lip_loss
            self.log("lip_loss", lip_loss, on_epoch=True)
        # L1 regularizer
        if self.l1 > 0.0:
            l1_loss = self.l1Reg(input_hat, target)
            loss = loss + self.l1 * l1_loss
            self.log("l1_loss", l1_loss, on_epoch=True)
            
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch: Tuple[tensor, tensor], batch_idx: int):
        input, target = batch
        input_hat = self(input)
        loss = self.criterion(input_hat, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    
    
    def test_step(self, batch: Tuple[tensor, tensor], batch_idx: int):
        input, target = batch
        pred = self(input)
        pred_unscaled = FusionData.rescale_label(pred, Fusion_label_mean, Fusion_label_std)
        target_unscaled = FusionData.rescale_label(target, Fusion_label_mean, Fusion_label_std)
        pred_ampl, pred_phase = FusionData.comp_to_ampl_phase(pred_unscaled[:,0], pred_unscaled[:,1])
        taregt_ampl, target_phase = FusionData.comp_to_ampl_phase(target_unscaled[:,0], target_unscaled[:,1])
        # used later 
        self.ampl_target.append(taregt_ampl)
        self.ampl_pred.append(pred_ampl)
        self.phase_target.append(target_phase)
        self.phase_pred.append(pred_phase)
        # compute the MAE
        ampl_mae = self.test_criterion(pred_ampl, taregt_ampl)
        phase_mea = self.test_criterion(pred_phase, target_phase)
        self.log("ampl_mae", ampl_mae)
        self.log("phase_mae", phase_mea)
        
    
    def on_test_epoch_end(self):
        # concatenate all the tensor coming from all the batches processed
        ampl_target = torch.cat(self.ampl_target, dim=0)
        phase_target = torch.cat(self.phase_target, dim=0)
        ampl_pred = torch.cat(self.ampl_pred, dim=0)
        phase_pred = torch.cat(self.phase_pred, dim=0)
        phase_target = phase_target * 180 / torch.pi
        phase_pred = phase_pred * 180 / torch.pi
        # store the amplitude and phase error distribution
        if hasattr(self.logger, "log_dir"):
            df = pd.DataFrame(np.vstack((ampl_target.detach().cpu().numpy(), 
                                         phase_target.detach().cpu().numpy(), 
                                         ampl_pred.detach().cpu().numpy(), 
                                         phase_pred.detach().cpu().numpy())).T, 
                                columns=['true_ampl', 'true_phase', 'pred_ampl', 'pred_phase'])
            pd.to_pickle(df, os.path.join(f"{self.logger.log_dir}/../", f"error_distribution.pkl"))

            # Apply filtering: Select elements of `true_ampl` between HIGH_AMPL_THRESHOLD and 30
        mask = (ampl_target >= 3) & (ampl_target <= 30)  # Boolean mask
        filtered_target_ampl = ampl_target[mask]
        filtered_pred_ampl = ampl_pred[mask]
        filtered_target_phase = phase_target[mask]
        filtered_pred_phase = phase_pred[mask]
        
        # Compute error
        error_ampl = filtered_pred_ampl - filtered_target_ampl
        error_phase = filtered_pred_phase - filtered_target_phase

        # Adjust phase error for the range [-180, 180]
        error_phase = torch.where(error_phase > 180, error_phase - 360, error_phase)
        error_phase = torch.where(error_phase < -180, error_phase + 360, error_phase)

        self.log_dict({
            "ampl_error_mean": torch.mean(error_ampl),
            "ampl_error_std": torch.std(error_ampl),
            "phase_error_mean": torch.mean(error_phase),
            "phase_error_std": torch.std(error_phase),
        })
        
        # clean the lists
        self.ampl_pred = []
        self.phase_pred = []
        self.ampl_target = []
        self.phase_target = []
        
        
        
        
    
    

            
        
        