import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -100)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_size = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True)
        self.drop_path = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=None, out_features=hidden_size, act_layer=nn.GELU, drop=0.)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GCN_s(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_s, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)  # Learnable weights
        self.ln = nn.LayerNorm(out_channels)  

    def forward(self, x_s, A_dim):
        x = torch.einsum('btd,dd->btd', x_s, A_dim)  # (batch_size, time_steps, out_channels)
        x = x.to(self.ln.weight.dtype)
        x = self.fc(x + x_s)
        # x = self.fc(x_s)
        x = torch.nn.functional.relu(x)
        x = self.ln(x)
        return x

class GCN_d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_d, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)  # Learnable weights
        self.ln = nn.LayerNorm(out_channels)  

    def forward(self, x_s, A_dim):
        x = torch.einsum('btd,dd->btd', x_s, A_dim)  # (batch_size, time_steps, out_channels)
        x = x.to(self.ln.weight.dtype)
        x = self.fc(x + x_s)
        # x = self.fc(x_s)
        x = torch.nn.functional.relu(x)
        x = self.ln(x)
        return x

class DaynamicFusion(nn.Module):
    def __init__(self, ni_dim=79, hidden_size=4, hidden_size_d=32, num_layers=3):
        super(DaynamicFusion, self).__init__()
        self.layer = num_layers
        self.hidden_size_d = hidden_size
        self.ni_dim = ni_dim
        self.ln = nn.LayerNorm(hidden_size_d)
        self.ln_ni = nn.LayerNorm(hidden_size_d)
        self.ni_fc = nn.Linear(self.ni_dim, hidden_size_d)
        self.ni_static = nn.Linear(hidden_size, hidden_size_d)
        self.ln_ni_static = nn.LayerNorm(hidden_size_d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size_d) * 0.02)
        max_len = 50
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, hidden_size_d, 2) * -(math.log(10000.0) / hidden_size_d))  # (dim/2,)
        self.gcn = GCN_d(ni_dim,ni_dim)
        pe = torch.zeros(max_len, hidden_size_d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_embedding', pe.unsqueeze(0))  # (1, max_len, dim)
        self.transformer_encoder = nn.ModuleList([Block(hidden_size_d) for i in range(self.layer)])

    def forward(self, ni, static, adj_matrix):
        b, l, ni_dim = ni.shape
        # gcn
        ni_proj = self.gcn(ni, adj_matrix)
        ni_proj = self.ln_ni(self.ni_fc(ni_proj))
        # mask
        ture_l = [(l-(ni[i].abs().sum(dim=-1) == 0).sum().item())+2 for i in range(b)]
        mask = torch.zeros(b, l+1, l+1).to(ni.device)
        for idx in range(b):
            mask[idx, :ture_l[idx], :ture_l[idx]] = 1
        # add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        ni_proj = torch.cat([cls_tokens, ni_proj], dim=1)
        # add position embedded
        ni_proj = ni_proj + self.positional_embedding[:,:l+1, :]
        if static == []:
            for i, m in enumerate(self.transformer_encoder, 1) :
                ni_proj = m(ni_proj, mask)

        else:
            transformer_encoder = self.transformer_encoder[0:len(static)]
            for i, (m, s) in enumerate(zip(transformer_encoder, static),1) :
                ni_proj = torch.cat([ni_proj, s], dim=1)
                
                ni_proj_np = ni_proj.detach().cpu().numpy() 
                np.save(f'ni_proj_layer_{i}.npy', ni_proj_np)
                mask = F.pad(mask, (0, 1, 1, 0), value=1)
                ni_proj = m(ni_proj, mask)
            
    
        dynamic_out = ni_proj[:, 0, :]
        return dynamic_out


class NoDaynamicFusion(nn.Module):
    def __init__(self, ni_dim=79, hidden_size=4, hidden_size_d=32, num_layers=3):
        super(NoDaynamicFusion, self).__init__()
        self.layer = num_layers
        self.hidden_size_d = hidden_size
        self.ni_dim = ni_dim
        self.ln = nn.LayerNorm(hidden_size_d)
        self.ln_ni = nn.LayerNorm(hidden_size_d)
        self.ni_fc = nn.Linear(self.ni_dim, hidden_size_d)
        self.ni_static = nn.Linear(hidden_size, hidden_size_d)
        self.ln_ni_static = nn.LayerNorm(hidden_size_d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size_d) * 0.02)
        max_len = 50
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, hidden_size_d, 2) * -(math.log(10000.0) / hidden_size_d))  # (dim/2,)
        self.gcn = GCN_d(ni_dim,ni_dim)
        pe = torch.zeros(max_len, hidden_size_d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_embedding', pe.unsqueeze(0))  # (1, max_len, dim)
        self.transformer_encoder = nn.ModuleList([Block(hidden_size_d) for i in range(self.layer)])

    def forward(self, ni, static, adj_matrix):
        b, l, ni_dim = ni.shape
        # gcn
        ni_proj = self.gcn(ni, adj_matrix)
        ni_proj = self.ln_ni(self.ni_fc(ni_proj))
        ni_proj = ni_proj.mean(dim=1)
        return ni_proj

class StaticFusion(nn.Module):
    def __init__(self, pe_dim, pi_dim, ls_dim, out_size=32):
        super(StaticFusion, self).__init__()
        self.pe_dim = pe_dim
        self.pi_dim = pi_dim
        self.ls_dim = ls_dim

        self.gcn_ls = GCN_s(self.ls_dim, out_size)
        self.gcn_pe = GCN_s(self.pe_dim, out_size)
        self.gcn_pi = GCN_s(self.pi_dim, out_size)
        self.gcn_fusion = GCN_s(self.ls_dim+self.pe_dim+self.pi_dim, out_size) 

    def forward(self, pe, pi, ls, adj_mrtrix):
        ls_scaled = adj_mrtrix['ls'].to(ls.device)
        pi_scaled = adj_mrtrix['pi'].to(ls.device)
        pe_scaled = adj_mrtrix['pe'].to(ls.device)
        f_scaled  =  adj_mrtrix['f'].to(ls.device)
        ls_gout   = self.gcn_ls(ls.unsqueeze_(1), ls_scaled)
        pe_gout   = self.gcn_pe(pe.unsqueeze_(1), pe_scaled)
        pi_gout   = self.gcn_pi(pi.unsqueeze_(1), pi_scaled)
        f = torch.cat([ls, pe, pi], dim=-1)
        f = self.gcn_fusion(f, f_scaled)
        return pe_gout, pi_gout, ls_gout

class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, args, epsilon=0.2):
        super().__init__()
        self.softlabel = args.softlabel
        self.epsilon = epsilon
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.softlabel:
            targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon  
            return self.loss_fn(logits, targets)
        else:
            return self.loss_fn(logits, targets)
    
class HealthPredictor(nn.Module):
    def __init__(self, args, hidden_size=16, hidden_size_d=32, num_layers=3, output_size=1):
        super(HealthPredictor, self).__init__()
        # ni 79 pe 16 pi 16 ls 10
        self.ni_dim = 79
        self.pe_dim = 16
        self.pi_dim = 16
        self.ls_dim = 10
        self.hidden_size = hidden_size
        self.hidden_size_d = hidden_size_d
        self.num_layers = num_layers
        # Static
        self.adj_matrix_s = torch.load('./data/adj_matrix2.pt')
        self.static_fusion = StaticFusion(self.pe_dim, self.pi_dim, self.ls_dim, out_size=hidden_size_d)
        # Dynamic
        self.adj_matrix = torch.load('./data/adj_matrix.pt').to(args.device)
        self.dynamic_fusion = DaynamicFusion(self.ni_dim, hidden_size, hidden_size_d, num_layers)
        # Classification
        self.fc = nn.Linear(hidden_size_d, output_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = BCEWithLogitsLossWithLabelSmoothing(args)
        self.relu = nn.ReLU()
        self._init_weights(self.fc)

    def forward(self, id, ls, pi, pe, ni, label):
        # ni 8,33,79
        # static
        b = id.shape[0]
        pe, pi, ls = self.static_fusion(pe, pi, ls, self.adj_matrix_s)
        # dynamic
        out = self.dynamic_fusion(ni, [pe, pi, ls], self.adj_matrix)
        # out = torch.cat([pe.squeeze(1), pi.squeeze(1), ls.squeeze(1)], dim=1)
        # fusion
        out = self.fc(out)
        pre = self.sigmoid(out.squeeze(1))
        loss = self.loss(out.squeeze(1), label.float())
        return pre, loss
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
if __name__ == "__main__":
    ni = torch.rand([8,33,79])
    # p = Pinjie()
    adj_matrix_s = torch.load('./data/adj_matrix.pt')
    pi = torch.randn([8,1,32])
    pe = torch.randn([8,1,32])
    ls = torch.randn([8,1,32])
    f = torch.randn([8,1,32])
    # dy = StaticFusion(16,16,10,48)
    # # x = x.reshape(32,144,-1)
    # out= dy(pi,pe,ls,adj_matrix_s)
    model = DaynamicFusion()
    out = model(ni,[pi,pe,ls,f],adj_matrix_s)
    print(out.shape)
    # print('r')
    # # r = r.reshape(32,72,134,-1)
    # w = WaveNet(134,0.2,7,1,144,32,32,128,64,4,72,2,2,32)
    # # r = r.permute(0,3,2,1)
    # rw = w(r)