import torch.nn as nn
import torch
import math
import torch.nn.functional as F

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
           q, k, v = qkv, qkv, qkv
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Improved mask handling logic, supporting different mask formats
        if mask is not None:
            # Detect mask shape and handle accordingly
            if mask.dim() == 2:  # mask with shape (B, N)
                # Expand dimensions to match attention matrix shape (B, num_heads, N, N)
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
                attn = attn.masked_fill(mask_expanded == 0, -1e3)
            elif mask.dim() == 3:  # mask with shape (B, N, N)
                # Expand dimensions to match attention matrix shape
                mask_expanded = mask.unsqueeze(1)  # (B, 1, N, N)
                attn = attn.masked_fill(mask_expanded == 0, -1e3)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Use contiguous to ensure memory continuity, use view instead of reshape for better dimension handling
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
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

class FeatureExpansion(nn.Module):
    def __init__(self, d_in, hidden_size_d):
        super().__init__()
        self.projection = nn.Linear(d_in, hidden_size_d)
    
    def forward(self, x):
        # x: (B, T, d_in)
        B, T, d_in = x.shape
        projected = self.projection(x)  # (B, T, hidden_size_d)
        output = x.unsqueeze(-1) * projected.unsqueeze(-2)  # (B, T, d_in, hidden_size_d)
        
        return output

class DaynamicFusion(nn.Module):
    def __init__(self, ni_dim=79, hidden_size=32, hidden_size_d=32, num_layers1=2, num_layers2=3):
        super(DaynamicFusion, self).__init__()
        # Args
        self.layer = num_layers2
        self.ni_dim = ni_dim
        # Input LayerNorm
        self.ln_ni = nn.LayerNorm(self.ni_dim)
        # Projection
        self.proj = FeatureExpansion(self.ni_dim, hidden_size_d)
        self.transformer_encoder1 = nn.ModuleList([Block(hidden_size_d) for i in range(num_layers1)])
        
        self.ni_fc = nn.Linear(self.ni_dim, hidden_size_d)
        self.relu = nn.ReLU()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size_d) * 0.02)
        max_len = 50
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, hidden_size_d, 2) * -(math.log(10000.0) / hidden_size_d))  # (dim/2,)
        pe = torch.zeros(max_len, hidden_size_d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_embedding', pe.unsqueeze(0))  # (1, max_len, dim)
        self.transformer_encoder2 = nn.ModuleList([Block(hidden_size_d) for i in range(self.layer)])

    def forward(self, ni):
        batch, l, ni_dim = ni.shape
        # LayerNorm
        
        mask = ~(ni.sum(dim=-1) == 0)
        # print(mask)
        ni = self.ln_ni(ni)# (B, T, d_in)
        # # Embedding
        # ni = self.proj(ni)  # (B, L, ni_dim, hidden_size_d)
        # # Intra-feature Modeling
        # ni = ni.reshape(batch*l, ni_dim, -1)
        # for i, b in enumerate(self.transformer_encoder1) :
        #     ni = b(ni)
        # ni = ni.reshape(batch, l, ni_dim, -1).mean(dim=2)
        # 
        # ni = self.ln_ni(ni)# (B, T, d_in)
        ni = self.relu(self.ni_fc(ni))
        # add cls token
        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # ni_proj = torch.cat([cls_tokens, ni_proj], dim=1)
        # add position embedded
        # ni_proj = ni_proj + self.positional_embedding[:,:l+1, :]
        for i, b in enumerate(self.transformer_encoder2) :
            ni_proj = b(ni, mask)
        dynamic_out = ni_proj
        return dynamic_out, mask

class ProgressiveFusion(nn.Module):
    def __init__(self, ni_dim=79, hidden_size=32, hidden_size_d=32, num_layers1=2, num_layers2=3):
        super(ProgressiveFusion, self).__init__()
        # Args
        self.layer = num_layers2
        self.ni_dim = ni_dim
        # Input LayerNorm
        self.ln_ni = nn.LayerNorm(self.ni_dim)
        # Projection
        self.proj = FeatureExpansion(self.ni_dim, hidden_size_d)
        self.transformer_encoder1 = nn.ModuleList([Block(hidden_size_d) for i in range(num_layers1)])
        
        self.ni_fc = nn.Linear(self.ni_dim, hidden_size_d)
        self.relu = nn.ReLU()
        self.transformer_encoder2 = nn.ModuleList([Block(hidden_size_d) for i in range(self.layer)])
    
    def create_causal_mask(self, seq_len, device):
        """
        Create a causal mask: lower triangular matrix (including diagonal) is True, rest is False.
        seq_len: sequence length
        device: device
        """
        # Create lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask
        
    def forward(self, ni, mask=None):
        batch, l, ni_dim = ni.shape
        # LayerNorm
        ni = self.ln_ni(ni)  # (B, T, d_in)

        ni = self.relu(self.ni_fc(ni))  # (B, T, hidden_size_d)
        seq_len = ni.shape[1]  # T
        causal_mask = self.create_causal_mask(seq_len, ni.device)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            batch_causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)  # (B, 1, T, T)
            combined_mask = mask_expanded & batch_causal_mask
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)  # (B, 1, T, T)

        x = ni
        for i, block in enumerate(self.transformer_encoder2):
            x = block(x, combined_mask)
        true_counts_per_batch = mask.sum(dim=1).tolist() 
        y=[]
        for i in range(batch):
            # print(x[i, :true_counts_per_batch[i], :].shape)
            y.append(x[i, :true_counts_per_batch[i], :].mean(dim=0))
        return torch.stack(y, dim=0)
        # return x.mean(dim=1)

class StaticFusion(nn.Module):
    def __init__(self, sta_dim=32, out_size=32, num_heads=4, sparsity=0.2):
        super(StaticFusion, self).__init__()
        self.sta_dim = sta_dim
        self.out_size = out_size
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.pos_embed = nn.Parameter(torch.randn(1, sta_dim, out_size) * 0.02)
        # Ensure out_size is divisible by sta_dim to support reshaping into d_static*D matrix
        assert out_size % sta_dim == 0, "out_size must be divisible by sta_dim"
        self.D = out_size // sta_dim  # Size of the second dimension after mapping
        
        self.learnable_token = nn.Parameter(torch.randn(1, 32, out_size) * 0.02)
        self.query_proj = nn.Linear(out_size, out_size)
        self.key_proj = nn.Linear(out_size, out_size)
        self.value_proj = nn.Linear(out_size, out_size)
        self.token_gate = nn.Linear(out_size * 2, 1)
        self.ln = nn.Linear(sta_dim, out_size)  # Map x_static to high-dimensional space
        self.ln2 = nn.Linear(self.D, out_size)  # Map x_static to high-dimensional space
        self.out_proj = nn.Linear(out_size, out_size)
        self.norm = nn.LayerNorm(out_size)
        self.relu = nn.ReLU()
    
    def forward(self, x_static):
        # Ensure input dimensions are correct
        if len(x_static.shape) == 2:
            x_static = x_static.unsqueeze(1)  # (B, 1, sta_dim)
        B = x_static.shape[0]
        
        # Map x_static to d_static*D matrix
        # 1. First, map to out_size dimension via linear layer
        x_proj = self.ln(x_static)  # (B, 1, out_size)
        
        # 2. Reshape to (batch_size, d_static, D), where d_static=sta_dim, D=out_size/sta_dim
        # Here we expand the sequence dimension so each static feature dimension has its own position
        x_reshaped = x_proj.reshape(B, self.sta_dim, self.D)  # (B, d_static, D)
         
        x_reshaped = self.ln2(x_reshaped)  # (B, 1, out_size)
        x_reshaped = x_reshaped + self.pos_embed  
        # print(x_reshaped.shape)
        # 3. For attention computation, treat it as a sequence, so swap dimensions
        # x_reshaped = x_reshaped.transpose(1, 2)  # (B, D, d_static)
        
        # 4. Re-project to out_size dimension for consistency
        x_proj = x_reshaped

        # Expand learnable_token to match new sequence length D
        token = self.learnable_token.expand(B, -1, -1)  # (B, D, out_size)
        
        # Compute gate
        gate_input = torch.cat([x_proj, token], dim=-1)  # (B, D, out_size*2)
        gate = torch.sigmoid(self.token_gate(gate_input))  # (B, D, 1)
        
        # Apply gate to token and add to x_proj
        q = x_proj + gate * token  # (B, D, out_size)
        
        # Compute attention
        q = self.query_proj(q)  # (B, D, out_size)
        k = self.key_proj(x_proj)  # (B, D, out_size)
        v = self.value_proj(x_proj)  # (B, D, out_size)
        
        # Multi-head attention
        head_dim = self.out_size // self.num_heads
        q = q.reshape(B, self.out_size, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, num_heads, D, head_dim)
        k = k.reshape(B, self.out_size, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, num_heads, D, head_dim)
        v = v.reshape(B, self.out_size, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, num_heads, D, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, num_heads, D, D)
        attn_scores_s = attn_scores
        # Sparse attention

        if self.sparsity > 0:
            top_k = max(1, int(attn_scores.shape[-1] * self.sparsity))
            values, indices = torch.topk(attn_scores, top_k, dim=-1)
            mask = torch.zeros_like(attn_scores)
            mask.scatter_(-1, indices, 1)
            attn_scores = attn_scores * mask - (1 - mask) * 1e9
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, D, D)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, D, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, self.out_size, self.out_size)  # (B, D, out_size)
        
        # Output projection and residual connection
        attn_output = self.out_proj(attn_output + x_proj)  # (B, D, out_size)
        
        # Aggregate information from all positions
        out = attn_output.mean(dim=1).unsqueeze(1)  # (B, 1, out_size)
        out = self.norm(out)  # (B, 1, out_size)
        out = self.relu(out)
        return out, (attn_scores, attn_scores_s)


class FBGPredictor(nn.Module):
    def __init__(self, hidden_size=32, hidden_size_d=64, num_layers1=1, num_layers2=2, 
                 cf_lambda=0.1, perturb_scale=0.05):
        super(FBGPredictor, self).__init__()
        # ni 79 pe 16 pi 16 ls 10
        self.static_dim = 32
        self.nut_dim = 79
        self.hidden_size = hidden_size
        self.hidden_size_d = hidden_size_d
        # Static
        self.static_fusion = StaticFusion(self.static_dim, out_size=hidden_size)
        # Dynamic
        self.dynamic_fusion = DaynamicFusion(self.nut_dim, hidden_size, hidden_size_d, num_layers1, num_layers2)
        # beta
        self.beta = nn.Sequential(
            nn.Linear(hidden_size, 16)
        )
        # Progressive Fusion
        self.progressive_fusion = ProgressiveFusion(self.hidden_size_d, hidden_size, hidden_size_d, num_layers1, num_layers2)
        # Prediction
        fusion_dim = hidden_size_d + hidden_size
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, 1) 
        )

        # Historical FBG processing module
        self.his_fbg_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        self.causal_conv = nn.Conv1d(16, 16, kernel_size=3, padding=2, bias=False)
        with torch.no_grad():
            weight = torch.zeros_like(self.causal_conv.weight)
            for i in range(weight.size(0)):
                for j in range(weight.size(1)):
                    for k in range(min(weight.size(2), i-j+1)):
                        weight[i, j, k] = 1.0
            self.causal_conv.weight = nn.Parameter(weight)
        
        self.alpha_module = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # ensure alpha is in [0, 1]
        )
        
        self.ln = nn.LayerNorm(self.static_dim)
        
        # Counterfactual causal inference parameters
        self.cf_lambda = cf_lambda  # weight of the counterfactual loss
        self.perturb_scale = perturb_scale  # scale of perturbation
        # epsilon for gradient estimation
        self.epsilon = 1e-6

    def forward(self, id, sta, ni, label):
        # Original input processing remains unchanged
        # history FBG
        his_fbg = ni[:,:,-1]
        # ni 8,33,79
        ni = ni[:,:,:-1]

        # static
        b = id.shape[0]
        h_sta, at = self.static_fusion(self.ln(sta))
        
        weights = self.beta(h_sta.squeeze(1))  # (B, T)
        weights = torch.sigmoid(weights) # (B, T)

        T = ni.shape[1]
        expanded_weights = weights.unsqueeze(-1)
        f_dynamic, mask = self.dynamic_fusion(ni)

        weighted_ni = f_dynamic * expanded_weights 
        
        # dynamic 
        out = self.progressive_fusion(weighted_ni, mask)
        out = torch.cat([h_sta.squeeze(1), out], dim=1)

        # fusion
        base_pred = self.reg_head(out).squeeze(1)
        
        # Process historical FBG and alpha
        true_counts_per_batch = mask.sum(dim=1).tolist()
        his_fbg_value = []
        alpha_values = []
        
        for i in range(his_fbg.shape[0]):
            if true_counts_per_batch[i] > 1:
                b_his_fbg = his_fbg[i, :true_counts_per_batch[i] - 1]  # exclude current time step FBG
                his_fbg_value.append(b_his_fbg[-1])
            else:
                b_his_fbg = []
                his_fbg_value.append(torch.tensor(0., device=base_pred.device))

            if len(b_his_fbg) > 0:
                fbg_feat = self.his_fbg_encoder(b_his_fbg.unsqueeze(-1))  # (seq_len, 16)

                fbg_feat = fbg_feat.transpose(0, 1).unsqueeze(0)  # (1, 16, seq_len)
                conv_out = self.causal_conv(fbg_feat)
                conv_out = conv_out.squeeze(0).transpose(0, 1)  # (seq_len, 16)

                last_feat = conv_out[-1]
                alpha = self.alpha_module(last_feat).squeeze()
                alpha_values.append(alpha)
            else:
                alpha_values.append(torch.tensor(0.5, device=base_pred.device))
        
        last_his_fbg = torch.stack(his_fbg_value, dim=0)
        alpha = torch.stack(alpha_values, dim=0)
        
        # Original prediction
        # out = alpha * last_his_fbg + (1 - alpha) * base_pred
        out = base_pred

        # Compute original MSE loss
        mse_loss = F.mse_loss(out, label.float())
        cf_loss = torch.tensor(0.0, device=mse_loss.device)
        # Counterfactual causal inference loss
        if self.training and self.cf_lambda > 0:

            with torch.enable_grad():
                perturbed_sta = sta.detach().clone().requires_grad_(True)
                
                # 1. The forward propagation yields the predicted result.
                p_h_sta, _ = self.static_fusion(self.ln(perturbed_sta))
                p_weights = torch.sigmoid(self.beta(p_h_sta.squeeze(1)))
                p_expanded_weights = p_weights.unsqueeze(-1)
                p_f_dynamic, _ = self.dynamic_fusion(ni.detach())
                p_weighted_ni = p_f_dynamic * p_expanded_weights
                p_out = self.progressive_fusion(p_weighted_ni, mask.detach())
                p_out = torch.cat([p_h_sta.squeeze(1), p_out], dim=1)
                p_base_pred = self.reg_head(p_out).squeeze(1)
                
                # 2. Calculate the gradient of the output with respect to the inpu
                gradients = torch.autograd.grad(
                    outputs=p_base_pred,
                    inputs=perturbed_sta,
                    grad_outputs=torch.ones_like(p_base_pred),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # 3. Calculate the norm of the gradient to identify the important directions
                gradient_norms = torch.norm(gradients, dim=-1, keepdim=True)
                
                # 4. Create small random perturbations
                perturbation = torch.randn_like(perturbed_sta) * self.perturb_scale
                proj_scale = torch.sum(perturbation * gradients, dim=-1, keepdim=True) / (gradient_norms.square() + self.epsilon)
                relevant_perturbation = proj_scale * gradients
                irrelevant_perturbation = perturbation - relevant_perturbation
                
                # Calculate counterfactual predictions
                counterfactual_sta = perturbed_sta + irrelevant_perturbation
                cf_h_sta, _ = self.static_fusion(self.ln(counterfactual_sta))
                cf_weights = torch.sigmoid(self.beta(cf_h_sta.squeeze(1)))
                cf_expanded_weights = cf_weights.unsqueeze(-1)
                cf_f_dynamic, _ = self.dynamic_fusion(ni.detach())
                cf_weighted_ni = cf_f_dynamic * cf_expanded_weights
                cf_out = self.progressive_fusion(cf_weighted_ni, mask.detach())
                cf_out = torch.cat([cf_h_sta.squeeze(1), cf_out], dim=1)
                cf_base_pred = self.reg_head(cf_out).squeeze(1)

                consistency_loss = F.mse_loss(p_base_pred, cf_base_pred)
                sparsity_loss = torch.mean(gradient_norms)
                cf_loss = consistency_loss + 0.1 * sparsity_loss
            
            # loss
            total_loss = mse_loss + self.cf_lambda * cf_loss
        else:
            total_loss = mse_loss
        
        return out, total_loss , mse_loss, cf_loss

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

if __name__ == "__main__":
    ni = torch.rand([8,16,79])
    # p = Pinjie()
    id= torch.randn([8])
    label=torch.randn([8])
    static = torch.randn([8,1,32])
    f = torch.randn([8,1,32])
    # dy = StaticFusion(16,16,10,48)
    # # x = x.reshape(32,144,-1)
    # out= dy(pi,pe,ls,adj_matrix_s)
    model = FBGPredictor()
    out, loss = model(id, static, ni, label)
    print(out.shape, loss)
    # print('r')
    # # r = r.reshape(32,72,134,-1)
    # w = WaveNet(134,0.2,7,1,144,32,32,128,64,4,72,2,2,32)
    # # r = r.permute(0,3,2,1)
    # rw = w(r)