#Import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorly as tl
from .. import cp_decomposition_ortho as ortho
#mport cp_decomposition_ortho as ortho
from tensorly.decomposition import parafac

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


class AddBiais(nn.Module):
    def __init__(self, biais):
        super(AddBiais, self).__init__()
        self.biais = nn.Parameter(biais)

    def forward(self, x):
        x = x.transpose(-2, -1) + self.biais
        return x.transpose(-2, -1)

    

def cp_decomposition_ortho(layer: nn.Module, rank: int, orthogonal=True, dim=0):
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
    if orthogonal:
        dec = ortho.cpDecomposition_help(weight_conv1D.numpy(), dim_ortho=dim, rank=rank)
        for i in range(len(dec)):
            dec[i] = torch.tensor(dec[i]).float()
        last, cin, kernel = dec
    else:
        dec = parafac(weight_conv1D.numpy(), rank=rank, init='random')
        for i in range(len(dec.factors)):
            dec.factors[i] = torch.tensor(dec.factors[i])
        last, cin, kernel = dec.factors

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


def cp_decomposition_ortho_list_concat(conv_layer_list: list, rank: int, orthogonal=True, dim=0):
    list_dim = [0]
    list_bias = []
    for indx, conv in enumerate(conv_layer_list):
        if indx == 0:
            stacked_tensor = conv.weight
            list_dim.append(conv.weight.shape[2])
            list_bias.append(conv.bias)
        else:
            stacked_tensor = torch.cat((stacked_tensor, conv.weight), dim=2)
            list_dim.append(list_dim[-1] + conv.weight.shape[2])
            list_bias.append(conv.bias)

    if orthogonal:
        dec = ortho.cpDecomposition_help(stacked_tensor.detach().numpy(), dim_ortho=dim, rank=rank)
        for i in range(len(dec)):
            dec[i] = torch.tensor(dec[i]).float()
        last, cin, kernel = dec
    else:
        dec = parafac(stacked_tensor.detach().numpy(), rank=rank, init='random')
        for i in range(len(dec.factors)):
            dec.factors[i] = torch.tensor(dec.factors[i])
        last, cin, kernel = dec.factors

    layer = conv_layer_list[0]
    depthwise_vertical_layer = \
        torch.nn.Conv2d(in_channels=cin.shape[1],
                        out_channels=cin.shape[1],
                        kernel_size=(cin.shape[0], 1), stride=1,
                        groups=cin.shape[1], bias=False)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(cin, 1, 0).unsqueeze(1).unsqueeze(-1)

    pointwise_r_to_t_layer = torch.nn.Conv1d(in_channels=last.shape[1],
                                            out_channels=last.shape[0], kernel_size=1, stride=1,
                                            padding=0, bias=False)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1)

    return_kernel = []
    for i, conv in enumerate(conv_layer_list):
        pointwise_horizontal_layer =\
        torch.nn.Conv2d(in_channels=1,
                    out_channels=conv.weight.shape[1], kernel_size=(1, conv.weight.shape[2]),
                    stride=layer.stride, padding=layer.padding, bias=False)
        pointwise_horizontal_layer.weight.data = \
            torch.transpose(kernel[list_dim[i]: list_dim[i + 1]], 1, 0).unsqueeze(1).unsqueeze(1)
        new_layers = [UnqueezeSecondDim(),
                pointwise_horizontal_layer,
                depthwise_vertical_layer,
                SqueezeThirdDim(),
                pointwise_r_to_t_layer,
                AddBiais(list_bias[i])]
        return_kernel.append(nn.Sequential(*new_layers))
    return return_kernel


def main():
   pass
    

if __name__ == "__main__":
    main()