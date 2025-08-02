import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConvRNNCell, DepthwiseSeparableConv2d

class ConvLSTMCell(BaseConvRNNCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, peephole=False, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_size, input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        self.peephole = peephole
        
        if peephole:
            self.Wci = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            self.Wcf = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            self.Wco = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))

        conv_classes = {
            'standard': nn.Conv2d,
            'depthwise': DepthwiseSeparableConv2d,
        }
        
        conv_class = conv_classes.get(conv_type, nn.Conv2d)
        conv_kwargs = {'kernel_size': kernel_size, 'padding': self.padding, 'bias': bias, 'dilation': dilation}
        
        self.conv = conv_class(input_dim + hidden_dim, 4 * hidden_dim, **conv_kwargs)
        self.reset_parameters()

    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.peephole:
            i = F.sigmoid(cc_i + self.Wci * c_prev)
            f = F.sigmoid(cc_f + self.Wcf * c_prev)
            o = F.sigmoid(cc_o + self.Wco * (f * c_prev + i * self.activation(cc_g)))
        else:
            i, f, o = F.sigmoid(cc_i), F.sigmoid(cc_f), F.sigmoid(cc_o)

        c_cur = f * c_prev + i * self.activation(cc_g)
        h_cur = o * self.activation(c_cur)
        return h_cur, c_cur

    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        state = (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                 torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        return (state[0].to(device), state[1].to(device)) if cuda else state

    def reset_parameters(self):
        super().reset_parameters()
        if self.peephole:
            for param in [self.Wci, self.Wcf, self.Wco]:
                param.data.uniform_(0, 1)

class ConvGRUCell(BaseConvRNNCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_size, input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        conv_classes = {
            'standard': nn.Conv2d,
            'depthwise': DepthwiseSeparableConv2d,
        }
        conv_class = conv_classes.get(conv_type, nn.Conv2d)
        
        conv_kwargs = {'kernel_size': kernel_size, 'padding': self.padding, 'bias': bias, 'dilation': dilation}
        
        self.conv_zr = conv_class(input_dim + hidden_dim, 2 * hidden_dim, **conv_kwargs)
        self.conv_h1 = conv_class(input_dim, hidden_dim, **conv_kwargs)
        self.conv_h2 = conv_class(hidden_dim, hidden_dim, **conv_kwargs)
        self.reset_parameters()

    def forward(self, input, h_prev):
        combined = torch.cat((input, h_prev), dim=1)
        z, r = torch.split(F.sigmoid(self.conv_zr(combined)), self.hidden_dim, dim=1)
        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))
        return (1 - z) * h_ + z * h_prev

    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        state = torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        return state.to(device) if cuda else state