import torch
from torch import nn
from brevitas.inject.enum import *
from brevitas.quant_tensor.torch_handler import quant_invariant_handler
from brevitas.quant import (
        Int8WeightPerTensorFloat, 
        Int8WeightPerChannelFloat, 
        Int8ActPerTensorFloat, 
        Uint8ActPerTensorFloat,
        IntBias)



class Unflatten(nn.Unflatten):
    def forward(self, x):
        return quant_invariant_handler(torch.unflatten, x, self.dim, self.unflattened_size)
    


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None



class CommonIntWeightPerChannelQuant(Int8WeightPerChannelFloat):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None



class CommonIntActQuant(Int8ActPerTensorFloat):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = RestrictValueType.FP



class CommonUintActQuant(Uint8ActPerTensorFloat):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    restrict_scaling_type = RestrictValueType.FP