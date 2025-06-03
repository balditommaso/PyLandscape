import torch
from typing import Dict, Optional
from torch.nn import Module
from typing import Dict, Optional

import torch.types
from collections import OrderedDict
from .curve_converter import curved_model
from .curve_module import Coeffs_t



class CurveNet(Module):
    """
    Module to handle all the curved modules.
    """
    def __init__(
        self, 
        curve: Module, 
        architecture: Module, 
        num_bends: int = 3, 
        fix_start: bool = True, 
        fix_end: bool = True,
    ) -> None:
        super(CurveNet, self).__init__()
        # prepare the masks for the bends
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        
        # instantiate the curve
        self.coeff_layer = curve(self.num_bends)
        # init coeffs
        coeffs_t = self.coeff_layer(0)
        Coeffs_t.value = coeffs_t
        # init the curved model
        self.net = curved_model(architecture, self.fix_points)
        

    def _get_model_by_index(self, index: int) -> Dict[str, torch.nn.Parameter]:
        res = OrderedDict()
        for name, param in self.net.named_parameters():
            value = name[-1]
            if value.isdigit() and int(value) == index:
                res[name] = param
                
        return res
     

    def import_base_parameters(self, base_model: Module, index: int) -> None:
        """
        Import the parameters into a specific band.
        """
        assert index in range(self.num_bends), "Index out of bound!"
        target_parameters =self._get_model_by_index(index)
        base_parameters = [param for name, param in base_model.named_parameters() if "activation" not in name]
        assert len(target_parameters) == len(base_parameters), "Models must have the same number of layer"
        for (name, target_param), base_param in zip(target_parameters.items(), base_parameters):
            target_param.data = base_param.data


    def init_linear(self) -> None:
        """
        Initialize the intermediate model in the line with the linear interpolation
        between the start and the end.
        """
        filtered_params = {name: param for name, param in self.net.named_parameters() if name[-1].isdigit()}
        band_dict = {}
        # build the list of version per layer
        for name, param in filtered_params.items():
            name, band = name.rsplit("_", 1)
            if name not in band_dict:
                band_dict[name] = [None] * self.num_bends
            band_dict[name][int(band)] = param 
        
        # apply linear interpolation to intermediate models
        for name, param_list in band_dict.items():
            for i in range (1, self.num_bends-1):
                alpha = i * 1.0 / (self.num_bends - 1)
                param_list[i].data.copy_(alpha * param_list[-1].data + (1.0 - alpha) * param_list[0].data)
        

    def forward(self, input: torch.tensor, t: Optional[float] = None) -> torch.tensor:
        # if t is not defined we generate random from uniform distribution
        if t is None:
            t = input.data.new(1).uniform_()
        # generate the coefficients
        coeffs_t = self.coeff_layer(t)
        Coeffs_t.value = coeffs_t
        output = self.net(input)
        return output

