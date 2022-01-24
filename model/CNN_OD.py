from torch import nn
import torch
from .decomposition_conv_ortho import cp_decomposition_ortho, cp_decomposition_ortho_list_concat

class CnnKim(nn.Module):
    ### CNN by kim
    def __init__(self, pretrained_matrix: torch.float = None, 
                 kernel_sizes: list = [3, 4, 5], nb_channel: int = 64,
                 one_embedding=True,
                 classification=True,name = 'kim_1d'):
        """[model]

        Args:
            pretrained_matrix (torch.float, optional): [description]. Defaults to None.
            kernel_sizes (list, optional): [Parameter to study n-gram]. Defaults to [3, 4, 5].
            nb_channel (int, optional): [nb channel for each conv1D]. Defaults to 64.
            one_embedding (bool, optional): [one or two embeddings like in kim]. Defaults to True.
            classification (bool, optional): [bool if we prefere classification or regression]. Defaults to True.
            name (str, optional): [place where save the model]. Defaults to 'kim_1d'.
        """
        super(CnnKim, self).__init__()
        #define parameter
        self.name = name
        self.one_embedding = one_embedding
        input_dim = pretrained_matrix.shape[1]
        assert (pretrained_matrix is not None), "Need a pretrained Matrix"
        
        #define embedding layer
        self.embedding = nn.Embedding(pretrained_matrix.shape[0], 
                                            pretrained_matrix.shape[1])
        self.const_embedding = nn.Embedding(pretrained_matrix.shape[0], 
                                            pretrained_matrix.shape[1])

        self.embedding.weight.data = pretrained_matrix
        if not(one_embedding):
            self.const_embedding.weight.data = pretrained_matrix
            self.const_embedding.weight.requires_grad = False
            input_dim *= 2
        
        #bottleneck for classificati or regression
        if classification:
            self.decoder = nn.Linear(len(kernel_sizes)*nb_channel, 2)
        else:
            self.decoder = nn.Sequential(nn.Linear(len(kernel_sizes)*nb_channel, len(kernel_sizes)*nb_channel),
                                          nn.Linear(len(kernel_sizes)*nb_channel, 1))
        self.dropout = nn.Dropout(0.5)
        
        #Conv block
        block = []
        for kernel_size in kernel_sizes:
            conv1d = nn.Conv1d(input_dim, nb_channel, kernel_size)
            component = nn.Sequential(
                conv1d, 
                nn.ReLU(), 
                nn.AdaptiveMaxPool1d(1))
            block.append(component)
        self.block = nn.ModuleList(block)
    
    def forward(self, inputs: torch.float):
        if not(self.one_embedding):
            embeddings = torch.cat((self.embedding(inputs), self.const_embedding(inputs)), dim = 2)
        else:
            embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        list_conv = [conv_block(embeddings) for conv_block in self.block]
        x = torch.cat(list_conv, -1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.decoder(x)
    
    def apply_cp_decomposition(self, rank: int=16, orthogonal=True, dim=0):
        """[Apply standard CP approach]

        Args:
            rank (int, optional): [rank for CP-decomposition]. Defaults to 16.
            orthogonal (bool, optional): [orthogonality or not]. Defaults to True.
            dim (int, optional): [which dim for orthogonality]. Defaults to 0.
        """
        if orthogonal:
            self.name = "kim_1d/ortho" + str(dim)
        else:
            self.name = "kim_1d/als"
        for i in range(len(self.block)):
            self.block[i][0] = cp_decomposition_ortho(self.block[i][0], rank, orthogonal, dim)
    
    def apply_cp_decomposition_fusion(self, rank: int=16, orthogonal=True, dim=0):
        """[Apply standard CP with fusion]

        Args:
            rank (int, optional): [rank for CP-decomposition]. Defaults to 16.
            orthogonal (bool, optional): [orthogonality or not]. Defaults to True.
            dim (int, optional): [which dim for orthogonality]. Defaults to 0.
        """
        if orthogonal:
            self.name = "kim_1d/ortho_fusion" + str(dim)
        else:
            self.name = "kim_1d/als_fusion"
        list_conv = []
        for i in range(len(self.block)):
            list_conv.append(self.block[i][0])
        
        list_conv_seq = cp_decomposition_ortho_list_concat(list_conv, rank, orthogonal, dim)
        for i in range(len(self.block)):
            self.block[i][0] = list_conv_seq[i]


def main():
    pass

if __name__ == '__main__':
    main()