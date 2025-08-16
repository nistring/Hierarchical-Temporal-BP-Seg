import torch
import torch.nn.functional as F
from .base import BaseConvRNNCell

class ConvLSTMCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim + hidden_dim, 4 * hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        c_cur = f * c_prev + i * self.activation(cc_g)
        h_cur = o * self.activation(c_cur)
        return h_cur, c_cur

    def init_hidden(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device))
        return state

class PeepholeConvLSTMCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim + hidden_dim, 4 * hidden_dim, conv_type, bias)
        self.conv_ci = self._get_conv_layer(hidden_dim, hidden_dim, conv_type, bias=False)
        self.conv_cf = self._get_conv_layer(hidden_dim, hidden_dim, conv_type, bias=False)
        self.conv_co = self._get_conv_layer(hidden_dim, hidden_dim, conv_type, bias=False)

        self.reset_parameters()

    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i + self.conv_ci(c_prev))
        f = torch.sigmoid(cc_f + self.conv_cf(c_prev))
        
        c_cur = f * c_prev + i * self.activation(cc_g)
        
        o = torch.sigmoid(cc_o + self.conv_co(c_cur))
        h_cur = o * self.activation(c_cur)
        
        return h_cur, c_cur

    def init_hidden(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device))
        return state

class ReducedConvLSTMCell(BaseConvRNNCell):
    # https://arxiv.org/pdf/1810.07251v5
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim + hidden_dim, hidden_dim * 2, conv_type, bias)
        self.w_cf = self._get_conv_layer(hidden_dim, hidden_dim, conv_type, bias=False)
        self.reset_parameters()

    def forward(self, input, prev_state):
        h, c = prev_state
        combined = torch.cat((input, h), dim=1)
        combined_conv = self.conv(combined)
        f, g = torch.split(combined_conv, self.hidden_dim, dim=1)
        f = torch.sigmoid(f + self.w_cf(c))
        g = self.activation(g)
        c = f * (c + g)
        h = self.activation(c) * f
        return h, c

    def init_hidden(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device))
        return state

class ConvGRUCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv_zr = self._get_conv_layer(input_dim + hidden_dim, 2 * hidden_dim, conv_type, bias)
        self.conv_h1 = self._get_conv_layer(input_dim, hidden_dim, conv_type, bias)
        self.conv_h2 = self._get_conv_layer(hidden_dim, hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, h_prev):
        combined = torch.cat((input, h_prev), dim=1)
        z, r = torch.split(torch.sigmoid(self.conv_zr(combined)), self.hidden_dim, dim=1)
        h_ = self.activation(self.conv_h1(input) + r * self.conv_h2(h_prev))
        return (1 - z) * h_ + z * h_prev

class ConvRNNCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim + hidden_dim, hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, h_prev):
        combined = torch.cat((input, h_prev), dim=1)
        h_cur = self.activation(self.conv(combined))
        return h_cur

class MinConvLSTMCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim, 3 * hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, h_prev):
        i, f, h_ = torch.split(self.conv(input), self.hidden_dim, dim=1)
        i, f = torch.sigmoid(i), torch.sigmoid(f)
        f_prime = f / (f + i)
        i_prime = i / (f + i)
        return f_prime * h_prev + i_prime * h_

class MinConvGRUCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv_zh = self._get_conv_layer(input_dim, 2 * hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, h_prev):
        z, h_ = torch.split(self.conv_zh(input), self.hidden_dim, dim=1)
        z = torch.sigmoid(z)
        return (1 - z) * h_ + z * h_prev
    
# https://arxiv.org/pdf/2508.03614v1
class MinConvExpLSTMCell(BaseConvRNNCell):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, conv_type='standard', dilation=1, **kwargs):
        super().__init__(input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.conv = self._get_conv_layer(input_dim, 3 * hidden_dim, conv_type, bias)
        self.reset_parameters()

    def forward(self, input, h_prev):
        i, f, h_ = torch.split(self.conv(input), self.hidden_dim, dim=1)
        f = torch.sigmoid(f - i)
        return f * h_prev + (1 - f) * h_