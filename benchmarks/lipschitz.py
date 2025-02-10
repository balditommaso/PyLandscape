import torch
import torch.nn as nn



def lipschitz_regularizer(model: nn.Module):
    """
    Compute the Parseval regularization penalty for a given weight matrix.
    
    Args:
        weight_matrix (torch.Tensor): The weight matrix to regularize.
        
    Returns:
        torch.Tensor: The Parseval regularization penalty.
    """
    
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            W = param.reshape(param.shape[0], -1)
            WW_t = torch.mm(W.t(), W)
            I = torch.eye(WW_t.size(0), device=W.device)
            loss += torch.norm(WW_t - I, p="fro").pow(2)
    return loss
    

