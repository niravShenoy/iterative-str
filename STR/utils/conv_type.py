from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
import numpy as np

DenseConv = nn.Conv2d

# def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
#     return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        return parser_args.sInit_value*torch.ones([1, 1])

class STRConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu

        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        else:
            self.sparseThreshold = nn.Parameter(initialize_sInit())
    
    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item()
    
    def getMask(self, f=torch.sigmoid):
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return temp
    
    def sparseFunction(self, x, s, activation=torch.relu, f=torch.sigmoid):
        return torch.sign(x)* activation(torch.abs(x)-f(s))

class STRConvER(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu
        # Static ER mask (never modified after initialization)
        self.er_mask = torch.zeros_like(self.weight).bernoulli_(p=parser_args.er_sparse_init)
        
        # Dynamic masks (modified during neuroregeneration)
        self.dynamic_prune_mask = torch.zeros_like(self.weight, dtype=torch.bool)  # Tracks pruned weights
        self.dynamic_growth_mask = torch.zeros_like(self.weight, dtype=torch.bool)  # Tracks regrown weights

        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        else:
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        
    def forward(self, x):
        # Calculate effective mask
        # effective_mask = ((self.er_mask.bool() & ~self.dynamic_prune_mask) | 
        #                  self.dynamic_growth_mask).float()

        effective_mask = (self.er_mask.bool() & ~self.dynamic_prune_mask).float()  
        
        # Apply STR and masking
        sparse_weight = self.sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        masked_weight = (effective_mask.to(sparse_weight.device) * sparse_weight).contiguous()
        
        return F.conv2d(
            x, masked_weight, self.bias, 
            self.stride, self.padding, 
            self.dilation, self.groups
        )
    
    def sparseFunction(self, x, s, activation=torch.relu, f=torch.sigmoid):
        return torch.sign(x)* activation(torch.abs(x)-f(s))
    
    def set_er_mask(self, p):
        self.er_mask = torch.zeros_like(self.weight).bernoulli_(p)

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        # effective_mask = ((self.er_mask.bool() & ~self.dynamic_prune_mask) | 
        #                  self.dynamic_growth_mask).float()
        effective_mask = (self.er_mask.bool() & ~self.dynamic_prune_mask).float() 
        temp = effective_mask * sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.sum().item(), f(self.sparseThreshold).item()
    
    def getMaskSparsity(self):
        # effective_mask = ((self.er_mask.bool() & ~self.dynamic_prune_mask) | 
        #                  self.dynamic_growth_mask).float()
        effective_mask = (self.er_mask.bool() & ~self.dynamic_prune_mask).float() 
        temp = effective_mask.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.sum().item()
    
    def getMask(self, f=torch.sigmoid):
        # effective_mask = ((self.er_mask.bool() & ~self.dynamic_prune_mask) |
        #                   self.dynamic_growth_mask).float()
        effective_mask = (self.er_mask.bool() & ~self.dynamic_prune_mask).float() 
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = effective_mask * sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return temp

class STRConv1d(STRConvER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        x = x.unsqueeze(-1).unsqueeze(-1)
        sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f).to(self.weight.device)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride
        )
        return x.squeeze()


class ConvER(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu
        self.er_mask = torch.zeros_like(self.weight, device=self.weight.device).bernoulli_(p=parser_args.er_sparse_init)
        
        
    def forward(self, x):
        
        sparseWeight = self.er_mask.to(self.weight.device) * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
    
    def sparseFunction(self, x, s, activation=torch.relu, f=torch.sigmoid):
        return torch.sign(x)* activation(torch.abs(x)-f(s))
    
    def set_er_mask(self, p):
        self.er_mask = torch.zeros_like(self.weight, device=self.weight.device).bernoulli_(p)

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = self.er_mask.to(self.weight.device) * self.weight
        temp = sparseWeight.detach().float()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), 0
    
    def getMaskSparsity(self):
        temp = self.er_mask.detach().float()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.sum().item()
    
    def getMask(self, f=torch.sigmoid):
        # sparseWeight = self.sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = (self.er_mask * self.weight.detach()).float()
        temp[temp!=0] = 1
        return temp

class ChooseEdges(autograd.Function):
    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class DNWConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print(f"=> Setting prune rate to {prune_rate}")

    def forward(self, x):
        w = ChooseEdges.apply(self.weight, self.prune_rate)

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x

def GMPChooseEdges(weight, prune_rate):
    output = weight.clone()
    _, idx = weight.flatten().abs().sort()
    p = int(prune_rate * weight.numel())
    # flat_oup and output access the same memory.
    flat_oup = output.flatten()
    flat_oup[idx[:p]] = 0
    return output

class GMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        self.curr_prune_rate = 0.0
        print(f"=> Setting prune rate to {prune_rate}")

    def set_curr_prune_rate(self, curr_prune_rate):
        self.curr_prune_rate = curr_prune_rate

    def forward(self, x):
        w = GMPChooseEdges(self.weight, self.curr_prune_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x