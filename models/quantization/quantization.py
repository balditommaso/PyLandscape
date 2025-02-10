import torch.nn as nn
from typing import Optional, Any, Dict
from collections import OrderedDict
import torch.nn as nn
import brevitas.nn as qnn
from copy import deepcopy
from brevitas import config
from brevitas.nn.quant_layer import WeightQuantType, ActQuantType
from brevitas.inject.enum import *
from .strategy import *
    
    
config.IGNORE_MISSING_KEYS = True



def get_quant_layer_config(
    weight_quant: WeightQuantType, 
    bit_width: int, 
    first_layer: bool,
    input_quant: ActQuantType
) -> Dict[str, Any]:
    """
    Retrieve the correct quantization for each module.

    Returns:
        Dict[str, nn.Module]: Dictionary with quantization information.
    """
    quant_config = dict(
        bias_quant=IntBias,
        weight_quant=weight_quant,
        weight_bit_width=bit_width
    )
    # add input quantization
    if first_layer:
        quant_config.update({
            "input_quant": input_quant,
            "input_bit_width": bit_width
        })

    return quant_config


def quantize_model(
    model: nn.Module, 
    precision: int,
    first_layer_bit_width: Optional[int] = None,
    act_bit_width: Optional[int] = None,
    input_quant: ActQuantType = CommonIntActQuant,
    fold_quant_input: bool = True,
    verbose: bool = True
) -> nn.Module:
    
    from pylandscape.mc_utils import rsetattr
    
    if act_bit_width is None:
        act_bit_width = precision
        
    first_layer = True
    q_model = deepcopy(model)
    
    # start the pipeline with an input quantizer
    if not fold_quant_input:
        first_layer = False
        q_module = OrderedDict([
            ('quant_input', qnn.QuantIdentity(
                    act_quant=input_quant,
                    bit_width=act_bit_width,
                    return_quant_tensor=True
                )
            )
        ])
        q_module.update(q_model._modules)
        q_model = nn.Sequential(q_module)
    
    # iterate over the module of the model and convert them to the quantized version
    for name, module in model.named_modules():
        
        # different precision for the first layer if needed
        bit_width = precision
        if first_layer and first_layer_bit_width is not None:
            bit_width = first_layer_bit_width
        
        q_module = None
        # ---------------------------------------------------------------------------- #
        #                                    Layers                                    #
        # ---------------------------------------------------------------------------- #
        if isinstance(module, nn.Conv2d):
            quant_config = get_quant_layer_config(
                CommonIntWeightPerChannelQuant,
                bit_width,
                first_layer,
                input_quant
            )
            q_module = qnn.QuantConv2d(
                module.in_channels, 
                module.out_channels, 
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
                **quant_config
            )
        elif isinstance(module, nn.ConvTranspose2d):
            quant_config = get_quant_layer_config(
                CommonIntWeightPerChannelQuant,
                bit_width,
                first_layer,
                input_quant
            )
            q_module = qnn.QuantConvTranspose2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
                output_padding=module.output_padding,
                **quant_config
            )
        elif isinstance(module, nn.Linear):
            out_features, in_features = module.weight.shape
            quant_config = get_quant_layer_config(
                CommonIntWeightPerTensorQuant,
                bit_width,
                first_layer,
                input_quant
            )
            q_module = qnn.QuantLinear(
                in_features,
                out_features,
                bias=module.bias is not None,
                **quant_config
            )
        # ---------------------------------------------------------------------------- #
        #                                  Activations                                 #
        # ---------------------------------------------------------------------------- #
        elif isinstance(module, nn.ReLU):
            q_module = qnn.QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                return_quant_tensor=True
            )
        elif isinstance(module, nn.Sigmoid):
            q_module = qnn.QuantSigmoid(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                return_quant_tensor=True
            )
        # ---------------------------------------------------------------------------- #
        #                                   Avg Pool                                   #
        # ---------------------------------------------------------------------------- #
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            q_module = qnn.TruncAdaptiveAvgPool2d(
                output_size=module.output_size,
                bit_width=bit_width
            )
        elif isinstance(module, nn.AvgPool2d):
            q_module = qnn.TruncAvgPool2d(
                kernel_size=module.kernel_size,
                stride=module.stride,
                bit_width=bit_width
            )
        # ---------------------------------------------------------------------------- #
        #                                    Special                                   #
        # ---------------------------------------------------------------------------- #
        elif isinstance(module, nn.Unflatten):
            q_module = Unflatten(module.dim, module.unflattened_size)
        else:
            # debug purpose
            if verbose:
                print(f"{module.__class__.__name__}:\t\t\tnot quantized")
            continue
        
        # load the parameters from the full precision version
        state_dict = module.state_dict()
        q_module.load_state_dict(state_dict)
        first_layer = False
        # substitute the module with the quantized version
        rsetattr(q_model, name, q_module)
    
    return q_model




