from copy import deepcopy
import torch
from torch import nn



def fold_bn_layers(model: nn.Module) -> nn.Module:
    '''
    Fold the 2D batch norm layer in the previous 2D convolution
    '''
    def _bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
        if conv_b is None:
            conv_b = bn_rm.new_zeros(bn_rm.shape)
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
        w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
        return nn.Parameter(w_fold), nn.Parameter(b_fold)
    
    def _fold_conv_bn_eval(conv, bn):
        # assert(not (conv.training or bn.training)), "Fusion only for eval!"
        fused_conv = deepcopy(conv)
        fused_conv.weight, fused_conv.bias = _bn_folding(fused_conv.weight, fused_conv.bias,
                                bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        return fused_conv
    
    
    new_model = deepcopy(model)
    new_model.eval()
    module_names = list(new_model._modules)
    for k, name in enumerate(module_names):
        if len(list(new_model._modules[name]._modules)) > 0:
            # iteratively re-apply the modifications
            new_model._modules[name] = fold_bn_layers(new_model._modules[name])
        else:
            if isinstance(new_model._modules[name], nn.BatchNorm2d):
                if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d):
                    # folded BN
                    folded_conv = _fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])
                    # replace old weight values and remove the BN layer
                    # new_model._modules.pop(name) # Remove the BN layer
                    new_model._modules[module_names[k]] = nn.Identity()
                    new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version
    new_model.train()
    return new_model