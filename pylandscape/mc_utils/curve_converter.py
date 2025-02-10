import brevitas.nn as qnn
import functools
from copy import deepcopy
from torch import nn
from torch.nn import Module
from typing import List
from .curve_module import Conv2d, Linear, ConvTranspose2D


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def curved_model(model: Module, fix_points: List[bool]) -> Module:
    """
    Method to convert a model to its curved version to shape the
    interpolation curve.

    Args:
        model (Module): Target model architecture.
        fix_points (List[bool]): Mask to block the training of 
                                 boundary models.

    Returns:
        Module: _description_
    """
    curve_model = deepcopy(model)
    # iterate over the module of the model and convert them to the quantized version
    for name, module in model.named_modules():
        module = deepcopy(module)
        curve_module = None
        if isinstance(module, nn.Conv2d) or isinstance(module, qnn.QuantConv2d):
            curve_module = Conv2d(module, fix_points)
        elif isinstance(module, nn.ConvTranspose2d):
            curve_module = ConvTranspose2D(module, fix_points)
        elif isinstance(module, nn.Linear) or isinstance(module, qnn.QuantLinear):
            curve_module = Linear(module, fix_points)
        else:
            continue
        
        rsetattr(curve_model, name, curve_module)
    return curve_model