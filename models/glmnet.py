import torch
import math
from torch import nn

# from EcaBlock import *
class FC_Block(nn.Module):
    def __init__(self, channels, ratio):
        super(FC_Block, self).__init__()
        # self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # if mode == "max":
        #     self.global_pooling = self.max_pooling
        # elif mode == "avg":
        #     self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
        self.sigmoid = nn.Sigmoid()
     
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # v = self.global_pooling(x).view(b, c)
        v = x.view(b,c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x*v

class MSFRF(nn.Module):
    
    def __init__(self, channels=32,r=4):
        super(MSFRF, self).__init__()
        inter_channels = int(channels // r) 
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.conv = nn.Conv1d(1, 1, kernel_size = 3, padding = (3 - 1) // 2, bias = False)
        self.bn = nn.BatchNorm2d(channels)
 
        self.attention2 = FC_Block(channels, 16)
 
        self.sigmoid = nn.Sigmoid()
 
 
    def forward(self, x, residual):
        x = x.unsqueeze(-1).unsqueeze(-1)
        residual = residual[:,-0,:].unsqueeze(-1).unsqueeze(-1)
        xa = x + residual

        xz1 = self.local_att1(xa)

        xz2 = self.conv(xa.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        xz2 = self.bn(xz2)

        xlg1 = xz2 + xz1
        xlg1 = self.attention2(xlg1)
        wei = self.sigmoid(xlg1)
 
 
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo.squeeze()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from models.model import StaticFusion, DaynamicFusion, BCEWithLogitsLossWithLabelSmoothing

class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims=(32,32,32), hidden_dims=(32,32,32), text_out= 2, dropouts=(0.1,0.1,0.1,0.1), output_dim=32, rank=4, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out= text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output


class HealthPredictor(nn.Module):
    def __init__(self, args, hidden_size=32, hidden_size_d=32, num_layers=3, output_size=1):
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
        self.lmf = MSFRF()
        self._init_weights(self.fc)

    def forward(self, id, ls, pi, pe, ni, label):
        # ni 8,33,79
        # static
        b = id.shape[0]
        pe, pi, ls = self.static_fusion(pe, pi, ls, self.adj_matrix_s)
        sta_f  = torch.stack([pe, pi, ls], dim=1)
        # print('s_f', sta_f.shape)
        # dynamic
        ni_f = self.dynamic_fusion(ni, self.adj_matrix)
        ni_f = ni_f.unsqueeze(1)  # [b, 32]
        # fusion
        out = self.lmf(pe.squeeze(), ni_f)
        # print('out', out.shape)
        out = self.fc(out)
        pre = self.sigmoid(out.squeeze(1))
        loss = self.loss(out.squeeze(1), label.float())
        # o = torch.concat(((pre-0.5).unsqueeze(1),(0.5-pre).unsqueeze(1)),1)
        # print('o', o.shape)
        return pre , loss
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


if __name__ == "__main__":

    model = MSFRF()#.cuda()
    print("Model loaded.")
    image = torch.rand(2, 32,1,1)#.cuda()
    audio = torch.rand(2, 32,1,1)#.cuda()
    print("Image and audio loaded.")

	# Run a feedforward and check shape
    c = model(image,audio)
    print(image.shape)
    print(c.shape)