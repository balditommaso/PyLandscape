import torch
from typing import Dict, Optional
from torch.nn import Module
from .curve_converter import curved_model
from .curve_module import CurveModule, Coeffs_t



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
        fix_end: bool = True
    ) -> None:
        super(CurveNet, self).__init__()
        # prepare the masks for the bends
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        
        # instantiate the curve
        self.coeff_layer = curve(self.num_bends)
        self.net = curved_model(architecture, fix_points=self.fix_points)
        
        # save a list with all the curved modules
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)
    
    
    @staticmethod
    def filter_weight_bias_parameters(
        named_params: Dict[str, torch.tensor],
        w_name: str = "weight",
        b_name: str = "bias"
    ) -> Dict[str, torch.nn.Parameter]:
        """
        Filters out parameters where the name does not contain 'weight' or 'bias'.
        
        Args:
            named_params: An iterable of (name, parameter) pairs, e.g., model.named_parameters()
            
        Returns:
            A dictionary containing only parameters where the name contains 'weight' or 'bias'.
        """
        filtered_params = {
            name: param for name, param in filter(lambda x: w_name in x[0] or b_name in x[0], named_params)
        }
        return filtered_params
     

    def import_base_parameters(self, base_model: Module, index: int) -> None:
        """
        Import the parameters into a specific band.
        """
        assert index in range(self.num_bends), "Index out of bound!"
        
        target_parameters = self.filter_weight_bias_parameters(
            self.net.named_parameters(), 
            f"weight_{index}", 
            f"bias_{index}"
        )
        base_parameters = self.filter_weight_bias_parameters(base_model.named_parameters())
        assert len(target_parameters) == len(base_parameters), "Models must have the same number of layer"
        param_dict = {}
        for name, param in target_parameters.items():
            param_dict[name[:-2]] = {'param': param}
        for name, param in base_parameters.items():
            param_dict[name]['base_param'] = param
        for _, param_obj in param_dict.items():
            param_obj['param'].data.copy_(param_obj['base_param'].data)


    def init_linear(self) -> None:
        """
        Initialize the intermediate model in the line with the linear interpolation
        between the start and the end.
        """
        filtered_param = self.filter_weight_bias_parameters(
            self.net.named_parameters(), 
            "weight_", 
            "bias_"
        )
        band_dict = {}
        # build the list of version per layer
        for name, param in filtered_param.items():
            name, band = name.rsplit("_", 1)
            if name not in band_dict:
                band_dict[name] = [None] * self.num_bends
            band_dict[name][int(band)] = param 
        
        # apply linear interpolation to intermediate models
        for name, param_list in band_dict.items():
            # print(f"init with linear interpolation layer: {name}")
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

