import pytorch_lightning as pl
from typing import Union
from brevitas import config
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from torchmetrics import Accuracy
from .quantization import *
from .utils import *
from benchmarks.lipschitz import lipschitz_regularizer as lipReg
from benchmarks.jacobian import JacobianReg as jReg
from .vgg import VGG_11, VGG_16
from .mobilenet import mobilenet_v1

config.IGNORE_MISSING_KEYS = True

AVAILABLE_MODELS = {
    "MobileNet": mobilenet_v1,
    "VGG11": VGG_11,
    "VGG16": VGG_16,
}

class VisionModel(pl.LightningModule):
    def __init__(self,
        config: Union[str, dict],
        architecture: str,
        quantized: bool, 
        learning_rate: float,
        bit_width: int = 32,
        pretrained: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        # load the config from yaml file
        if isinstance(config, str):
            config = yaml_load(config)
        
        self.learning_rate = learning_rate
        if quantized: 
            self.learning_rate *= 0.01
            
        self.quantized = quantized
        self.save_hyperparameters()
        self.bit_width = bit_width
        
        if architecture not in AVAILABLE_MODELS:
            raise ValueError(f"The model {config['save_dir']} is not yet available!")
        
        self.model = AVAILABLE_MODELS[architecture](config['data']['num_classes'])
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config["fit"]["label_smoothing"]
        )
        
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
    
    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.l2)
        scheduler = None
        if self.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        elif self.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min')
        elif self.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=200)
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
        x.requires_grad = True 
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
        
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        # first, load the state_dict
        is_quantized = self.hparams["bit_width"] < 32
        needs_transformation = "model.features.init_block.bn.weight" in state_dict
        # load full precision model 
        if not is_quantized:
            return super().load_state_dict(state_dict, strict)
        # load quantized model
        if is_quantized and needs_transformation:
            super().load_state_dict(state_dict, strict)
            config = self.hparams["config"]["model"]["quantization"]
            self.model = fold_bn_layers(self.model)
            self.model = apply_quantization(self.model, self.bit_width, config)
        else:
            config = self.hparams["config"]["model"]["quantization"]
            self.model = fold_bn_layers(self.model)
            self.model = apply_quantization(self.model, self.bit_width, config)
            return super().load_state_dict(state_dict, strict)

            




