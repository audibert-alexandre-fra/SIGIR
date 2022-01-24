#Import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorly as tl
from .import cp_decomposition_ortho as ortho


class UnqueezeSecondDim(nn.Module):
    def __init__(self):
        super(UnqueezeSecondDim, self).__init__()
    
    def forward(self, x):
        return x.unsqueeze(1)

    
class SqueezeThirdDim(nn.Module):
    def __init__(self):
        super(SqueezeThirdDim, self).__init__()
    
    def forward(self, x):
        return x.squeeze(2)
    

def cp_decomposition_ortho(layer: nn.Module, rank: int):
    """[transform on conv1D into conv2D/conv2D/conv1D with cp decompostion]

    Args:
        layer ([nn.Module]): [conv1D]
        rank ([int]): [the rank of the cp-decomposition]

    Returns:
        [nn.Module]: [conv2D, conv2D, conv2D]
    """
    unsqueeze_second = UnqueezeSecondDim()
    squeeze_third = SqueezeThirdDim()
    weight_conv1D = layer.weight.data
    dec = ortho.cpDecomposition_help(weight_conv1D.numpy(), dim_ortho=0, rank=rank)
    for i in range(len(dec)):
        dec[i] = torch.tensor(dec[i]).float()
    last, cin, kernel = dec

    pointwise_horizontal_layer =\
    torch.nn.Conv2d(in_channels=1,
                    out_channels=kernel.shape[1], kernel_size=(1, kernel.shape[0]),
                    stride=layer.stride, padding=layer.padding, bias=False)

    depthwise_vertical_layer = \
        torch.nn.Conv2d(in_channels=cin.shape[1],
                        out_channels=cin.shape[1],
                        kernel_size=(cin.shape[0], 1), stride=1,
                        groups=cin.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv1d(in_channels=last.shape[1],
                                            out_channels=last.shape[0], kernel_size=1, stride=1,
                                            padding=0, bias=True)

    #Creation de biais
    pointwise_r_to_t_layer.bias.data = layer.bias.data

    #On met les poids obtenus precedement au bon format!
    pointwise_horizontal_layer.weight.data = \
        torch.transpose(kernel, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(cin, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1)

    new_layers = [unsqueeze_second,
                pointwise_horizontal_layer,
                  depthwise_vertical_layer,
                  squeeze_third,
                  pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)


def main():
    # decomposition conv2D
    layer = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=4)
    layer_2 = cp_decomposition_ortho(layer, 16)
    test = torch.rand(1, 20, 10)
    a = layer_2(test)
    print(a.shape)
    

if __name__ == "__main__":
    main()