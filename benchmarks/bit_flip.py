from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.types import _device
from pylandscape import Hessian



class BitFlip:
    """
    Class used to simulate the bit-flip stress test on
    models quantized with Brevitas.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        dataloader: DataLoader = None, 
        device: _device = torch.device("cpu")
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_bits = 0
        self.quant_info = {}
        self.indices = None
        
        prev_num_params = 0
        for name, layer in self.model.named_modules():
            if hasattr(layer, "quant_weight"):
                
                quant_weight = layer.quant_weight()
                num_parameters = torch.numel(layer.weight)
                
                self.quant_info[name] = {
                    "num_params": num_parameters,
                    "bit_width": int(quant_weight.bit_width.item()),
                    "first": prev_num_params,
                    "last": prev_num_params + num_parameters
                }
                self.num_bits += int(quant_weight.bit_width.item() * num_parameters)
                prev_num_params += num_parameters
                       
                       
    def from_param_ranking_to_bit_ranking(self, param_ranking):
        """
        This method convert the param ranking to a bit ranking giving higher score to the MSB
        """
        last_bit_idx = -1
        param_to_bit_indices_dict = dict()
        # Build a dictionary such that (key, val) = param_index, ([bit_indices])
        for wi in range(len(param_ranking)):
            bit_indices_associated_with_param = []
            
            bit_width = None
            for info in self.quant_info.values():
                if wi >= info["first"] and wi < info["last"]:
                    bit_width = info["bit_width"]
            
            for wbi in reversed(range(bit_width)):
                last_bit_idx += 1
                bit_indices_associated_with_param.append((last_bit_idx, wbi))

            param_to_bit_indices_dict[wi] = bit_indices_associated_with_param

        # Add the bit indices into the appropriate list based on
        # the parameter ranking
        max_bit_width = max(self.quant_info.values(), key=lambda x: x["bit_width"])["bit_width"]
        sorted_msb_lsb_lists = [[] for _ in range(max_bit_width)]
        for param_idx in param_ranking.tolist():
            for bit_idx, wbi in param_to_bit_indices_dict[param_idx]:
                sorted_msb_lsb_lists[wbi].append(bit_idx)

        merged_msb_to_lsb_list = []
        for l in sorted_msb_lsb_lists:
            merged_msb_to_lsb_list.extend(l)

        return merged_msb_to_lsb_list
            
    
    @staticmethod
    def dequantize(value: int, scale: float, zero_point: float) -> float:
        """
        Dequantize the integer value to bring it back in the float representation.
        """
        return (value - zero_point) * scale
    
    
    @staticmethod
    def integer_signed_bitflip(value: torch.Tensor, bit_width: int, index: int) -> int:
        """
        Flip a bit of a specified signed integer value.
        """
        assert index >= 0 and index < bit_width, \
               f"Bit index out of bound!, index={index} bit_width=[{bit_width},0]"
               
        max_value = (1 << bit_width) - 1
        flipped_value = torch.bitwise_xor(value, 1 << index)
        flipped_value = torch.bitwise_and(flipped_value, max_value)
        sign_bit = 1 << (bit_width - 1)
        if torch.bitwise_and(flipped_value, sign_bit):
            flipped_value -= (1 << bit_width)
        
        return flipped_value
    
    
    @staticmethod
    def integer_unsigned_bitflip(value: int, bit_width: int, index: int) -> int:
        """
        Flip a bit of a specified unsigned integer value.
        """
        assert index >= 0 and index < bit_width, \
               f"Bit index out of bound!, index={index} bit_width=[{bit_width},0]"
        max_value = (1 << bit_width) - 1
        flipped_value = torch.bitwise_xor(value, 1 << index)
        flipped_value = torch.bitwise_and(flipped_value, max_value)
        
        return flipped_value
                        
        
    def attack(self, num: int, strategy: str) -> nn.Module:
        """
        Method used to flip a certain number of bits in the target layers of a neural network.

        Args:
            num (int): Number of bit to be flipped
            strategy (str, optional): How the number of bits are chose.

        Raises:
            ValueError: If the number of bits < 0.

        Returns:
            nn.Module: Model with the bits flipped.
        """
        assert num >= 0, "The number of bit to flip cannot be negative!"
        
        # get the indices if necessary
        if strategy == "random":
            self.indices = torch.randint(0, self.num_bits, size=(self.num_bits,)).tolist()
        elif strategy == "fkeras":
            hessian = Hessian(self.model, self.model.criterion, self.dataloader)
            param_ranking, _ = hessian.hessian_rank()
            self.indices = self.from_param_ranking_to_bit_ranking(param_ranking)
        else:
            raise ValueError(f"Not valid strategy: {strategy}")
        
        # copy the model
        perturbed_model = deepcopy(self.model)
        
        # flip the bits
        for index in self.indices[:num]:
            acc = 0
            for name, layer in perturbed_model.named_modules():
                # looking for quantized layer only
                if hasattr(layer, "quant_weight"):
                    bit_width = self.quant_info[name]["bit_width"]
                    weight_index = int(index // bit_width)
                    bit_index = int(index % bit_width)
                    
                    if weight_index < self.quant_info[name]["num_params"] + acc:
                        weight_index -= acc # align the index to the layer
                        weight = layer.quant_weight().int().reshape(-1)[weight_index]
                        if layer.quant_weight().signed:
                            # flip a bit in the signed integer
                            flipped_value = BitFlip.integer_signed_bitflip(weight, bit_width, bit_index)
                        else:
                            # flip a bit in the unsigned integer
                            flipped_value = BitFlip.integer_unsigned_bitflip(weight, bit_width, bit_index)
                        
                        # handling layer-wise and channel-wise quantization
                        scale = layer.quant_weight().scale
                        if scale.numel() > 1:
                            scale = scale[weight_index // (layer.kernel_size[0]*layer.kernel_size[1]*layer.in_channels)]
                        zero_point = layer.quant_weight().zero_point
                        if zero_point.numel() > 1:
                            zero_point = zero_point[weight_index // (layer.kernel_size[0]*layer.kernel_size[1])]
                            
                        # apply the bit-flip
                        with torch.no_grad():
                            flipped_weight = BitFlip.dequantize(flipped_value, scale, zero_point)
                            layer.weight.reshape(-1)[weight_index] = flipped_weight
                        
                        break
                    else:
                        acc += self.quant_info[name]["num_params"]
                    
        return perturbed_model


