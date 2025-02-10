import torch
from torch import nn, tensor
from typing import Union, Tuple, Mapping, Any
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
from itertools import starmap
from datamodule.telescope_loss.utils_pt import unnormalize, emd 
from collections import OrderedDict
from .utils import yaml_load
from .telescope_pt import telescopeMSE8x8
from datamodule.autoencoder_datamodule import ARRANGE, ARRANGE_MASK
from benchmarks.jacobian import JacobianReg as jReg
from benchmarks.lipschitz import lipschitz_regularizer as lipReg
from .quantization import CommonIntActQuant, CommonUintActQuant, quantize_model



CALQ_MASK = torch.tensor(
    [
        1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 0, 0, 0, 0,
    ]
)


ENCODER_SIZE = {
    "baseline": (3, 8, 128),
    "large": (5, 32, 288),
    "small": (3, 1, 16),
}


def econ_model(size: str = "baseline") -> Tuple[nn.Module, nn.Module]:
    """_summary_

    Args:
        size (str): Models size available (small, baseline, large). Defaults to "baseline".

    Returns:
        Tuple(nn.Module, nn.Module): Encoder and Decoder part of the model.
    """
    ENCODED_DIM = 16
    INPUT_SHAPE = (1, 8, 8)  # PyTorch defaults to (C, H, W)
    kernel_size, num_kernels, fc_input = ENCODER_SIZE[size]

    # build the encoder
    encoder = nn.Sequential(OrderedDict([
        ("conv2d", nn.Conv2d(
                in_channels=1, 
                out_channels=num_kernels, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1
            )
        ),
        ("relu1", nn.ReLU()),
        ("flatten", nn.Flatten()),
        ("dense", nn.Linear(fc_input, ENCODED_DIM)),
    ]))    
    # build the decoder    
    decoder = nn.Sequential(OrderedDict([
            ("dec_dense", nn.Linear(ENCODED_DIM, 128)),
            ("relu1", nn.ReLU()),
            ("unflatten", nn.Unflatten(1, (8, 4, 4))),
            ("convtrans2d1", nn.ConvTranspose2d(
                    in_channels=8, 
                    out_channels=8, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                )
            ),
            ("relu2", nn.ReLU()),
            ("convtrans2d2", nn.ConvTranspose2d(
                    in_channels=8, 
                    out_channels=INPUT_SHAPE[0], 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                )
            ),
            ("sigmoid", nn.Sigmoid()),
        ]))
    
    return encoder, decoder



class AutoEncoder(pl.LightningModule):
    def __init__(
        self, 
        config: Union[str, dict],
        quantized: bool,
        learning_rate: float, 
        bit_width: int = 32,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.INPUT_SHAPE = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None
        
        if isinstance(config, str):
            config = yaml_load(config)
        
        self.save_hyperparameters()
        self.quantized = quantized
        self.learning_rate = learning_rate

        self.encoder, self.decoder = econ_model(config["model"]["size"])

        # quantize the model
        if self.quantized:
            quant_cfg = config['model']['quantization']
            # pick the right input quantization
            input_quant = CommonUintActQuant
            if hasattr(quant_cfg, "input_quant") and \
                quant_cfg["input_quant"] != "CommonUintActQuant":
                input_quant = CommonIntActQuant
            # prepare the quantization configuration of the model
            quant_cfg = dict(
                first_layer_bit_width=quant_cfg.get('first_layer_bit_width', None),
                act_bit_width=quant_cfg.get('act_bit_width', None),
                input_quant=input_quant,
                verbose=quant_cfg.get('verbose', False),
            )
            # quantize the encoder
            self.encoder = quantize_model(self.encoder, bit_width, fold_quant_input=False, **quant_cfg)
        
        self.criterion = telescopeMSE8x8
        self.input_calQ = []
        self.output_calQ = []
        
        self.scheduler = config['fit']['lr_scheduler']
        # regularizer
        self.l1 = config['fit']['regularizer'].get('l1', 0.0)
        self.l2 = config['fit']['regularizer'].get('l2', 0.0)
        self.jacobian = config['fit']['regularizer'].get('jacobian', 0.0)
        self.parseval = config['fit']['regularizer'].get('parseval', 0.0)
        
        self.jReg = jReg(n=1)
        self.lipReg = lipReg
        self.l1Reg = nn.L1Loss()

            
    @property
    def model(self) -> nn.Module:
        return nn.Sequential(self.encoder, self.decoder)
    
    
    def invert_arrange(self):
        """
        Invert the arrange mask
        """
        remap = []
        hashmap = {}  # cell : index mapping
        found_duplicate_charge = len(ARRANGE[ARRANGE_MASK == 1]) > len(
            torch.unique(ARRANGE[ARRANGE_MASK == 1])
        )
        for i in range(len(ARRANGE)):
            if ARRANGE_MASK[i] == 1:
                if found_duplicate_charge:
                    if CALQ_MASK[i] == 1:
                        hashmap[int(ARRANGE[i])] = i
                else:
                    hashmap[int(ARRANGE[i])] = i
        for i in range(len(torch.unique(ARRANGE))):
            remap.append(hashmap[i])
        return torch.tensor(remap)


    def map_to_calq(self, x):
        """
        Map the input/output of the autoencoder into CALQs orders
        """
        remap = self.invert_arrange()
        image_size = self.INPUT_SHAPE[0] * self.INPUT_SHAPE[1] * self.INPUT_SHAPE[2]
        reshaped_x = torch.reshape(x, (len(x), image_size))
        reshaped_x[:, ARRANGE_MASK == 0] = 0
        return reshaped_x[:, remap]


    def set_val_sum(self, val_sum):
        self.val_sum = val_sum


    def predict(self, x):
        decoded_Q = self(x)
        encoded_Q = self.encoder(x)
        encoded_Q = torch.reshape(encoded_Q, (len(encoded_Q), self.encoded_dim, 1))
        return decoded_Q, encoded_Q


    def forward(self, x: tensor):
        return self.decoder(self.encoder(x))


    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.l2
        )  
        scheduler = None
        if self.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                patience=5, 
                factor=0.1, 
                min_lr=1e-6, 
                threshold=0.01, 
                cooldown=3
            )
        elif self.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=3, last_epoch=-1)
        elif self.scheduler == 'one_cycle':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                three_phase=True,
                steps_per_epoch=1
            )
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
            enc_lip_loss = self.lipReg(self.encoder)
            dec_lip_loss = self.lipReg(self.decoder)
            loss = loss + self.parseval * (enc_lip_loss + dec_lip_loss)
            self.log("lip_loss", enc_lip_loss + dec_lip_loss, on_epoch=True)
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
        output = self(input)
        input_calQ = self.map_to_calq(target)
        output_calQ_fr = self.map_to_calq(output)
        input_calQ = torch.stack(
            [input_calQ[i] * self.val_sum[i] for i in range(len(input_calQ))]
        )  
        output_calQ = unnormalize(torch.clone(output_calQ_fr), self.val_sum)  
        self.input_calQ.append(input_calQ)
        self.output_calQ.append(output_calQ)
    
    
    def on_test_epoch_end(self):
        # concatenate all the tensor coming from all the batches processed
        input_calQ = torch.cat(self.input_calQ, dim=0)
        output_calQ = torch.cat(self.output_calQ, dim=0)
        self.input_calQ = []
        self.output_calQ = []
        # compute the average EMD
        emd_list = torch.Tensor(list(starmap(emd, zip(input_calQ, output_calQ))))
        average_emd = emd_list.nanmean().item()
        result = {'AVG_EMD': average_emd}
        self.log_dict(result, sync_dist=True)
        return result
    
    
    
    
    