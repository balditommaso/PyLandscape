import numpy as np
import torch
from . import Metric
from copy import deepcopy
from torch import tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from warnings import warn
from scipy.stats import entropy
from tqdm import tqdm




class NeuralEfficiency(Metric):
    def __init__(
        self,
        model: Module = None,
        dataloader: DataLoader = None,
        name: str = "neural_efficiency",
        performance: str = None,
        max_batches: int = None
    ) -> None:
        super().__init__(model, dataloader, name)
        self.performance = performance  # used to compute the aIQ
        self.results = {}   # there will be different values
        self.max_batches = max_batches if max_batches else len(dataloader)
        self.activations = {}
        self.model = self.model_wrapper(model)
        
        
    def model_wrapper(self, model: Module) -> Module:
        
        def hook_fn(module: Module, input: tensor, output: tensor) -> None:
            self.activations[module] = output
        
        model = deepcopy(model)
        for module in model.modules():
            if hasattr(module, "weight"):
                self.activations[module] = 0
                module.register_forward_hook(hook_fn)
                
        return model

    
    @staticmethod       
    def quantize_activation(x: tensor):
        """
        We are interested only if the neuron fired (>0) or not (=0)
        """
        return torch.where(x > 0, 1, 0)
    
    
    def get_outputs(self, activation):
        """
        the convolutional layer will produce more outputs per input (one per channel),
        so we will iterate over each channel and return an array with all the outputs.
        """
        assert activation.shape[0] == 1, \
            f"The batch size should be 1 instead of {activation.shape[0]}!"
            
        outputs = []
        # removing the batch size
        activation = torch.squeeze(activation, dim=0)
        if len(activation.shape) == 3:
            # 2dconv
            for channel_idx in range(activation.shape[0]):
                channel_output = activation[channel_idx, :, :]
                outputs.append(channel_output)
        elif len(activation.shape) == 1:
            # dense (or similar)
            outputs.append(activation)
        else:
            warn("Warning: activation dimension not handled yet!")
        return outputs
    
    
    def entropy_per_layer(self, layers):
        # do not train the network
        self.model.eval()
        #iterate over the batches to compute the probability of each state
        state_space = {}
        counter = 0
        with tqdm(self.dataloader, unit="batch") as tepoch:
            tepoch.set_description("Test efficiency")
            for batch, _ in self.dataloader:
                counter += 1
                
                # get the activations of each layer
                self.model(batch)
                for name, act in self.activations.items():
                    # handle possible problems with the tensors
                    if act is None:
                        warn(f"Attention: the layer {name} has None features!")
                        continue

                    # the convolutional layer will produce more outputs per input (one per channel),
                    # so I will iterate over each channel
                    outputs = self.get_outputs(act)
                    # each output state must be quantized and we record its frequency
                    for output_state in outputs:
                        # dictionary to record the probabilities
                        # quantize the activations (convert to bytes to use it as key)
                        quant_activation = self.quantize_activation(output_state).detach().cpu().numpy().tobytes()
                        # record the probabilities for each layer
                        if name not in state_space:
                            state_space[name] = {"num_neurons": torch.numel(output_state)}
                        if quant_activation not in state_space[name]:
                            state_space[name][quant_activation] = 1
                        else:
                            state_space[name][quant_activation] += 1
                if counter >= self.max_batches:
                    break
        for name, state_freq in state_space.items():
            # compute the probabilities for each output state
            probabilities = []
            num_outputs = sum(state_freq.values())
            for freq in state_freq.values():
                probabilities.append(freq / num_outputs)
            # compute the entropy of the layer
            probabilities = np.array(probabilities)
            layer_entropy = entropy(probabilities, base=2)
            state_space[name]["entropy"] = layer_entropy
        return state_space
    
    
    def compute(self, beta=2):
        """
        Compute the neural efficiency metrics.
        """
        # compute the entropy for each layer
        entropies = self.entropy_per_layer(self.target_layers)
        # compute neuron efficiency for each layer
        layers_efficiency = {}
        for name, layer in entropies.items():
            layers_efficiency[name] = layer["entropy"] / layer["num_neurons"]
        # compute network efficiency, which is the geometric mean of the efficiency
        # of all the layers
        network_efficiency = 1
        for efficiency in layers_efficiency.values():
            network_efficiency *= efficiency
        network_efficiency = network_efficiency ** (1/len(layers_efficiency.items()))
        #compute the aIQ, which is a combination between neural network efficiency and model performance
        aIQ = None
        if self.performance is not None:
            aIQ = ((self.performance ** beta) * network_efficiency) ** (1 / (beta + 1))
        else:
            warn("Warning: you cannot compute the aIQ without the performance of the model (accuracy, EMD, MSE, ...).")

        self.results = {
            "layers_efficiency": layers_efficiency,
            "network_efficiency": network_efficiency,
            "aIQ": aIQ
        }
        return self.results