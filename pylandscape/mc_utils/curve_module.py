import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torch.nn import Module, Parameter
from typing import Tuple, List, Callable



class Coeffs_t:
    """
    Global variable t shared by all the curved modules
    """
    value = 0


class CurveWeightComputation(Module):
    """
    Parametrization used to update the parameters of Brevitas 
    in order to keep the gradients during the quantization process.
    """
    def __init__(self, weight_computation: Callable, index: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_computation = weight_computation
        self. index = index
        
        
    def forward(self, weight: torch.tensor) -> torch.tensor:
        coeffs_t = Coeffs_t.value
        return self.weight_computation(coeffs_t)[self.index]



class CurveModule(Module):
    """
    Class used as base to be extended by all curved version of other modules. 
    """
    def __init__(self, fix_points: List[bool], parameter_names: Tuple[str] = ()) -> None:
        # super(CurveModule, self).__init__()
        Module.__init__(self)
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
                parameter = getattr(self, f"{parameter_name}_{j}")
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
        return w_t



class QuantLinear(CurveModule):
    
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("weight", "bias"))

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
        # add parametrization of the weight
        register_parametrization(
            self.module, "weight", CurveWeightComputation(self.compute_weights_t, 0)
        )
        self.module.parametrizations.weight.original.requires_grad = False
        register_parametrization(
            self.module, "bias", CurveWeightComputation(self.compute_weights_t, 1)
        )   
        self.module.parametrizations.bias.original.requires_grad = False
        
        
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
        return self.module(input)


class QuantConv2d(CurveModule):
    
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("weight", "bias"))

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
        # add parametrization of the weight
        register_parametrization(
            self.module, "weight", CurveWeightComputation(self.compute_weights_t, 0)
        )
        self.module.parametrizations.weight.original.requires_grad = False
        register_parametrization(
            self.module, "bias", CurveWeightComputation(self.compute_weights_t, 1)
        )
        self.module.parametrizations.bias.original.requires_grad = False
        
        
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
        return self.module(input)


class QuantNLAL(CurveModule):
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("value",))
        value = module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value.data
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                f"value_{i}",
                Parameter(torch.Tensor(value), requires_grad=not fixed)
            )

        self.module = module
        # add parametrization of the weight
        register_parametrization(
            self.module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl, 
            "value", 
            CurveWeightComputation(self.compute_weights_t, 0)
        )
        self.module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.parametrizations.value.original.requires_grad = False
        
        
    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.module(input)



class Linear(CurveModule):
    """
    Curved version of the `nn.Linera` module.
    """
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("weight", "bias"))
        self.out_features, self.in_features = module.weight.shape
        
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(torch.Tensor(
                    self.out_features, self.in_features), requires_grad=not fixed
                )
            )
            if module.bias is not None:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(self.out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter("bias_%d" % i, None)
        
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Reset the parameter of the weights and bias as done in PyTorch
        """
        stdv = 1. / math.sqrt(self.in_features)
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
        
        return F.linear(input, weight_t, bias_t)
        


class Conv2d(CurveModule):
    """
    Curved version of the `nn.Conv2d` module.
    """
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("weight", "bias"))
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        
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
     
        self.reset_parameters()
        

    def reset_parameters(self) -> None:
        """
        Reset the layer parameters with the initial values as done in PyTorch
        """
        n = self.in_channels
        for k in self.kernel_size:
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

        return F.conv2d(
            input, 
            weight_t, 
            bias_t, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )



class ConvTranspose2D(CurveModule):
    def __init__(self, module: nn.Module, fix_points: List[bool]) -> None:
        CurveModule.__init__(self, fix_points, ("weight", "bias"))
        self.stride = module.stride
        self.padding = module.padding
        self.output_padding = module.output_padding
        self.groups = module.groups
        self.dilation = module.dilation
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        
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

        self.reset_parameters()
        
        
    def reset_parameters(self):
        """
        Reset the layer parameters with the initial values as done in PyTorch
        """
        n = self.in_channels
        for k in self.kernel_size:
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
            
        return F.conv_transpose2d(
            input, 
            weight_t, 
            bias_t, 
            self.stride, 
            self.padding, 
            self.output_padding,
            self.groups,
            self.dilation
        )