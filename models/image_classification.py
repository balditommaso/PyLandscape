import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union
from brevitas import config
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from torchmetrics import Accuracy
from .quantization import *
from .utils import *
from benchmarks.lipschitz import lipschitz_regularizer as lipReg
from benchmarks.jacobian import JacobianReg as jReg


config.IGNORE_MISSING_KEYS = True


REPO_LINK = 'chenyaofo/pytorch-cifar-models'


def available_models():
    return torch.hub.list(REPO_LINK, force_reload=True)


def get_model(model:str, pretrained:bool = True) -> nn.Module:
    """_summary_

    Args:
        model (str): name of the model as stored in the repo `chenyaofo/pytorch-cifar-models`
        pretrained (bool, optional): Flag to download the trained parameters. Defaults to True.

    Returns:
        nn.Module: _description_
    """
    model = torch.hub.load(REPO_LINK, model, pretrained=pretrained)
    if model.__class__.__name__ == 'RepVGG':
        model.convert_to_inference_model()

    return model


class VisionModel(pl.LightningModule):
    def __init__(self,
            config: Union[str, dict],
            quantized: bool, 
            learning_rate: float,
            bit_width: int=32,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # load the config from yaml file
        if isinstance(config, str):
            config = self.yaml_load(config)
        
        self.learning_rate = learning_rate
        self.quantized = quantized
        self.save_hyperparameters()
        
        # load the model from the repo
        model_arch = config['model']['name']
        pretrained = config['model']['pretrained']
        
        self.model = get_model(model_arch, pretrained=pretrained)
        if self.quantized:
            self.model = quantize_model(self.model, bit_width, **config['model']['quantization'])
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = config['fit']['lr_scheduler']
        # regularizers
        self.l1 = config['fit']['regularizer'].get('l1', 0.0)
        self.l2 = config['fit']['regularizer'].get('l2', 0.0)
        self.jacobian = config['fit']['regularizer'].get('jacobian', 0.0)
        self.parseval = config['fit']['regularizer'].get('parseval', 0.0)
        
        self.jReg = jReg(n=1)
        self.lipReg = lipReg
        self.l1Reg = nn.L1Loss()
        
        # define the metrics
        self.train_top1_acc = Accuracy(task='multiclass', top_k=1, num_classes=config['data']['num_classes'])
        self.train_top5_acc = Accuracy(task='multiclass', top_k=5, num_classes=config['data']['num_classes'])
        self.val_top1_acc = Accuracy(task='multiclass', top_k=1, num_classes=config['data']['num_classes'])
        self.val_top5_acc = Accuracy(task='multiclass', top_k=5, num_classes=config['data']['num_classes'])
        self.test_top1_acc = Accuracy(task='multiclass', top_k=1, num_classes=config['data']['num_classes'])
        self.test_top5_acc = Accuracy(task='multiclass', top_k=5, num_classes=config['data']['num_classes'])
        
        
    def yaml_load(self, config):
        with open(config) as stream:
            param = yaml.safe_load(stream)
        return param
    
    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.l2)
        scheduler = None
        if self.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min')
        elif self.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=3)
        elif self.scheduler == 'one_cycle':
            scheduler = OneCycleLR(optimizer, 
                                   max_lr=self.learning_rate,
                                   epochs=self.trainer.max_epochs,
                                   three_phase=True,
                                   steps_per_epoch=1)
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
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True # this is essential!
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # apply regularizer
        if self.jacobian > 0.0:
            j_loss = self.jReg(x, y_hat)
            loss = loss + self.jacobian * j_loss
        if self.parseval > 0.0:
            lip_loss = self.lipReg(self.model)
            loss = loss + self.parseval * lip_loss
        if self.l1 > 0.0:
            l1_loss = self.l1Reg(y_hat, y)
            loss = loss + self.l1 * l1_loss
        
        self.train_top1_acc(y_hat, y)
        self.train_top5_acc(y_hat, y)
        
        self.log('train_top1_acc', self.train_top1_acc, prog_bar=True)
        self.log('train_top5_acc', self.train_top5_acc)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_top1_acc(y_hat, y)
        self.val_top5_acc(y_hat, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_top1_acc', self.val_top1_acc, prog_bar=True, sync_dist=True)
        self.log('val_top5_acc', self.val_top5_acc, sync_dist=True)
        
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_top1_acc(y_hat, y)
        self.test_top5_acc(y_hat, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_top1_acc', self.test_top1_acc, prog_bar=True, sync_dist=True)
        self.log('test_top5_acc', self.test_top5_acc, sync_dist=True)
        



