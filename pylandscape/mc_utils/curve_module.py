import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from copy import deepcopy
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
from typing import Tuple, List, Union, Optional, Dict
import brevitas.nn as qnn



class Coeffs_t:
    """
    Global variable t shared by all the curved modules
    """
    value = 0



class CurveModule(Module):
    """
    Class used as base to be extended by all curved version of other modules. 
    """
    def __init__(self, fix_points: List[bool], parameter_names: Tuple[str] = ()) -> None:
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names


    def compute_weights_t(self, coeffs_t: List[float]) -> List[torch.tensor]:
        """
        Method used to compute the configurations of parameters along the curve at given `t`.

        Args:
            coeffs_t (List[float]): t coefficient to sample from the curve.

        Returns:
            List[torch.tensor]: List of model"s parameters.
        """
        w_t = [None] * len(self.parameter_names)
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, "%s_%d" % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
        return w_t



class Linear(CurveModule):
    """
    Curved version of the `nn.Linera` module.
    """
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        super(Linear, self).__init__(fix_points, ("weight", "bias"))

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(torch.Tensor(
                    module.out_features, module.in_features), requires_grad=not fixed
                )
            )
            if module.bias is not None:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(module.out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter("bias_%d" % i, None)
            
        self.module = module
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Reset the parameter of the weights and bias as done in PyTorch
        """
        stdv = 1. / math.sqrt(self.module.in_features)
        for i in range(self.num_bends):
            getattr(self, "weight_%d" % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, "bias_%d" % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)


    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Compute the weights based on the coefficients of the curve
        """
        # compute the weights
        coeffs_t = Coeffs_t.value
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        
        # manually set the weights and bias in the quant layer
        self.module.weight.data = weight_t
        if self.module.bias is not None:
            self.module.bias.data = bias_t

        return self.module(input)



class Conv2d(CurveModule):
    """
    Curved version of the `nn.Conv2d` module.
    """
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        super(Conv2d, self).__init__(fix_points, ("weight", "bias"))
        
        if module.in_channels % module.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if module.out_channels % module.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(
                    torch.Tensor(
                        module.out_channels, 
                        module.in_channels // module.groups, 
                        *module.kernel_size
                    ),
                    requires_grad=not fixed
                )
            )
            if module.bias is not None:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(module.out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter("bias_%d" % i, None)
     
        self.module = module
        self.reset_parameters()
        

    def reset_parameters(self) -> None:
        """
        Reset the layer parameters with the initial values as done in PyTorch
        """
        n = self.module.in_channels
        for k in self.module.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, "weight_%d" % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, "bias_%d" % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)


    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Compute the weights based on the coefficients of the curve
        """
        # compute the weights based on the curve
        coeffs_t = Coeffs_t.value
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        
        # manually set the weights and bias in the quant layer
        self.module.weight.data = weight_t
        if self.module.bias is not None:
            self.module.bias.data = bias_t

        return self.module(input)



class ConvTranspose2D(CurveModule):
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        super(ConvTranspose2D, self).__init__(fix_points, ("weight", "bias"))
        
        if module.in_channels % module.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if module.out_channels % module.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(
                    torch.Tensor(
                        module.in_channels, 
                        module.out_channels // module.groups, 
                        *module.kernel_size
                    ),
                    requires_grad=not fixed
                )
            )
            if module.bias is not None:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(module.out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter("bias_%d" % i, None)

        self.module = module
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """
        Reset the layer parameters with the initial values as done in PyTorch
        """
        n = self.module.in_channels
        for k in self.module.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, "weight_%d" % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, "bias_%d" % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)
                
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Compute the weights based on the coefficients of the curve
        """
        coeffs_t = Coeffs_t.value
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        
        # manually set the weights and bias in the quant layer
        self.module.weight.data = weight_t
        if self.module.bias is not None:
            self.module.bias.data = bias_t

        return self.module(input)
