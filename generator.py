#定义encoder子模块，用于生成缺失数据
class Encoder8(nn.Module):
    """
    输入真实缺失数据X、邻接矩阵A
    定义一个GAN的模型+gcn+Atten
    """
    def __init__(self,input_size,input_dim,hid_dim,hid_dim1,out_dim):
        super(Encoder8, self).__init__()
        #隐藏层设置为节点数
        self.input_size = input_size
        self.cnn_block = CNN_BRAN_ATTEN(input_size)
        self.gcn1 = GCN_encoder1(input_dim, hid_dim)
        self.gcn2 = GCN_encoder2(hid_dim, hid_dim1)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hid_dim1, hid_dim1 // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hid_dim1 // 2, hid_dim1 // 4),
            torch.nn.Linear(hid_dim1 // 4, out_dim)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=1, stride=1),
        )

    def forward(self, adj,x,SCM,M):
        x = x.masked_fill(M, 0)
        x1 = self.cnn_block(x)
        for i in range(self.input_size):
            h1 = F.relu(self.gcn1(adj, x[i:i + 1, :].T, SCM))
            h2 = F.relu(self.gcn2(adj, h1, SCM))
            out = self.MLP(h2)
            if i == 0:
                h3 = out
            else:
                h3 = torch.cat((h3, out), dim=1)
        x3 = h3.T + x1
        output = self.conv1(x3)
        return output

