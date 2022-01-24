import torch
from torch import nn

def permutation(n: int):
    return list(range(1, n)) + [0]

class CrossStitchUnits(nn.Module):
    def __init__(self, layer_1: nn.Module, layer_2: nn.Module, init: float=1):
        super(CrossStitchUnits, self).__init__()
        self.out_dim = layer_2.weight.shape[0]
        self.type = None
        if type(layer_1) != type(layer_2):
            assert "Type layer is different"
        if layer_1.weight.shape != layer_2.weight.shape:
            assert " Error dimension"        
        if isinstance(layer_1, nn.Linear):
            self.type = 'linear'
        if isinstance(layer_1, nn.Conv2d):
            self.type = 'conv2D'
        if isinstance(layer_1, nn.Conv1d):
            self.type = 'conv1D'
            
        weight_1 = torch.tensor([[init, 1-init]]*layer_2.weight.shape[0])
        weight_2 = torch.tensor([[1-init, init]]*layer_2.weight.shape[0])
        self.mat_1 = nn.Parameter(weight_1.float())
        self.mat_2 = nn.Parameter(weight_2.float())
        
    def forward(self, x1, x2):
        data = torch.stack((x1, x2))
        data = data.permute(permutation(data.ndim))
        if self.type == "linear":
            res_1 = data*self.mat_1
            res_2 = data*self.mat_2
            res_1 = res_1.sum(-1)
            res_2 = res_2.sum(-1)
        elif self.type == "conv2D":
            data = data.permute(0, 2, 3, 1, 4)
            print(data.shape)
            print(self.mat_1.shape)
            res_1 = data*self.mat_1
            res_2 = data*self.mat_2
            res_1 = res_1.sum(-1)
            res_2 = res_2.sum(-1)            
            res_1 = res_1.permute(0, 3, 1, 2)
            res_2 = res_2.permute(0, 3, 1, 2)
        elif self.type == "conv1D":
            data = data.permute(0, 2, 1, 3)
            res_1 = data*self.mat_1
            res_2 = data*self.mat_2
            res_1 = res_1.sum(-1)
            res_2 = res_2.sum(-1)
            res_1 = res_1.permute(0, 2, 1)
            res_2 = res_2.permute(0, 2, 1)            
        return res_1, res_2
    
    def initilisation_cross(self, init):
        weight_1 = torch.tensor([[init, 1-init]]*self.out_dim)
        weight_2 = torch.tensor([[1-init, init]]*self.out_dim)
        self.mat_1.data = weight_1
        self.mat_2.data = weight_2

def main():
    a = nn.Conv1d(2, 3, kernel_size=3)
    b = nn.Conv1d(2, 3, kernel_size=3)
    cross = CrossStitchUnits(a, b, init=0.75)
    data = torch.rand((1, 2, 5))
    x1 = a(data)
    x2 = b(data)
    x1[0][1][2] = 1
    x2[0][1][2] = 0
    res_1, res_2 = cross(x1, x2)
    print(res_1)
    print(res_2)
    

if __name__ == '__main__':
    main()