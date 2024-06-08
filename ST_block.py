#定义STTransformer中的SA Block和TA Block
class SSelfAttention(nn.Module):
    '空间self-attention'
    def __init__(self, embed_size, heads):
        super(SSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N, T, C = query.shape

        values = values.reshape(N, T, self.heads, self.head_dim)
        keys   = keys.reshape(N, T, self.heads, self.head_dim)
        query  = query.reshape(N, T, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("qthd,kthd->qkth", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.einsum("qkth,kthd->qthd", [attention, values]).reshape(
            N, T, self.heads * self.head_dim
        )
        out = self.fc_out(out)

        return out
    
class TSelfAttention(nn.Module):
    '时间self-attention'
    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N, T, C = query.shape

        values = values.reshape(N, T, self.heads, self.head_dim)
        keys   = keys.reshape(N, T, self.heads, self.head_dim)
        query  = query.reshape(N, T, self.heads, self.head_dim)

        values  = self.values(values)  # (N, T, heads, head_dim)
        keys    = self.keys(keys)      # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("nqkh,nkhd->nqhd", [attention, values]).reshape(
                N, T, self.heads * self.head_dim
        )
        out = self.fc_out(out)

        return out
    
    
class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = nn.Parameter(adj)
        self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        
        self.attention = SSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        # 调用GCN
        self.gcn = GCN(embed_size, embed_size*2, embed_size, dropout)  
        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
                
        # Spatial Embedding 部分
        N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)
        
        
        # GCN 部分
        X_G = torch.Tensor(query.shape[0], 0, query.shape[2])
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        
        for t in range(query.shape[1]):
            o = self.gcn(query[ : , t,  : ],  self.adj)
            o = o.unsqueeze(1)
            X_G = torch.cat((X_G, o), dim=1)

        # Spatial Transformer 部分
        query = query+D_S
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))
        
        # 融合 STransformer and GCN
        g = torch.sigmoid( self.fs(U_S) +  self.fg(X_G) )
        out = g*U_S + (1-g)*X_G

        return out
    
class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding
        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)

        self.branch3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,2), padding=(1,1), output_padding=(0,1),
                               bias=False),

            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
                      ),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                      ),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
                      )
        )
        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t,x_p):
        N, T, C = query.shape
        # print(query.shape)
        D_T = self.temporal_embedding(torch.arange(0, T))
        D_T = D_T.expand(N, T, C)
        x_p = self.branch2(x_p)
        x_p=self.branch3(x_p)

        x_p = x_p.permute(1, 2, 0)
        query = query + D_T +x_p
        
        attention = self.attention(value, key, query)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


    
    
    