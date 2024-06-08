#定义GCN、ATTEN等模块
class GCN_encoder(nn.Module):
    '定义图卷积类，处理邻接矩阵A和特征序列X和掩膜矩阵M'
    def __init__(self, input_dim,output_dim, use_bias=True):
        super(GCN_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(1, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adj, x,SCM):
        "邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"
        input_feature=x
        support = torch.mm(input_feature.to(torch.float32), self.weight)
        adj1= torch.sparse.mm(adj, SCM.to(torch.float32))
        output = torch.mm(adj1, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

class Spatial_atten(nn.Module):
    '定义空间注意力'
    def __init__(self,input_dim,output_dim):
        super(Spatial_atten,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight1 = nn.Parameter(torch.Tensor(24, input_dim))
        self.sigma=nn.ReLU()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight1)

    def forward(self,x):
        x1 = torch.mul(x, self.weight1)
        x1=self.sigma(x1)
        out=x1+x
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'
