# Adopted from https://github.com/aserdega/convlstmgru/blob/master/convlstm.py and https://github.com/aserdega/convlstmgru/blob/master/convgru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Base classes for code reuse
class BaseConvRNNCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, activation=F.tanh, dilation=1):
        super().__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = ((kernel_size[0] - 1) * dilation) // 2, ((kernel_size[1] - 1) * dilation) // 2
        self.bias = bias
        self.activation = activation

    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        raise NotImplementedError

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    module.bias.data.zero_()

class BaseConvRNN(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, activation=F.tanh, dilation=1, cell_class=None, **cell_kwargs):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        
        # Extend parameters for multilayer
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation = self._extend_for_multilayer(activation, num_layers)
        dilation = self._extend_for_multilayer(dilation, num_layers)

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create cell list
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i-1]
            cell_list.append(cell_class(
                input_size=(self.height, self.width),
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim[i],
                kernel_size=kernel_size[i],
                bias=bias,
                activation=activation[i],
                dilation=dilation[i],
                **cell_kwargs
            ))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.reset_parameters()

    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list)
            and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# Convolution variants
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=padding, groups=in_channels, bias=False, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class BaseConvTTCell(BaseConvRNNCell):
    """Base class for Convolutional Tensor-Train cells"""
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, order=3, steps=3, ranks=8, dilation=1):
        super().__init__(input_size, input_dim, hidden_dim, kernel_size, bias, activation, dilation)
        
        self.order = order
        self.steps = steps
        self.ranks = ranks
        self.lags = steps - order + 1
        self.hidden_states = None
        self.hidden_pointer = 0
        
        # Temporal processing cores (shared between LSTM and GRU)
        Conv3d = lambda in_channels, out_channels: nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, bias=bias,
            kernel_size=kernel_size + (self.lags,), padding=self.padding + (0,))
        
        self.layers_ = nn.ModuleList([Conv3d(hidden_dim, ranks) for _ in range(order)])
        
    def _get_conv2d(self, in_channels, out_channels):
        """Helper to create Conv2d with consistent parameters"""
        return nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, 
                        padding=self.padding, bias=self.bias, dilation=self.dilation)
    
    def _process_temporal_states(self):
        """Common temporal processing logic"""
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps
            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = torch.stack(input_states[:self.lags], dim=-1)
            input_states = torch.squeeze(self.layers_[l](input_states), dim=-1)
            
            if l == 0:
                temp_states = input_states
            else:
                temp_states = input_states + self._get_spatial_output(l-1, temp_states)
        return temp_states
    
    def _get_spatial_output(self, layer_idx, temp_states):
        """Get spatial layer output - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _update_hidden_buffer(self, outputs):
        """Update hidden states buffer"""
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps

class ConvTTLSTMCell(BaseConvTTCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, order=3, steps=3, ranks=8, dilation=1):
        super().__init__(input_size, input_dim, hidden_dim, kernel_size, bias, activation, dilation, order, steps, ranks)
        
        self.cell_states = None
        
        # Spatial processing layers for LSTM gates
        self.layers = nn.ModuleList()
        for l in range(order):
            in_channels = ranks if l < order - 1 else ranks + input_dim
            out_channels = ranks if l < order - 1 else 4 * hidden_dim
            self.layers.append(self._get_conv2d(in_channels, out_channels))
        
        self.reset_parameters()
    
    def _get_spatial_output(self, layer_idx, temp_states):
        return self.layers[layer_idx](temp_states)
    
    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        self.hidden_states = [torch.zeros(batch_size, self.hidden_dim, self.height, self.width) 
                             for _ in range(self.steps)]
        self.cell_states = torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        self.hidden_pointer = 0
        
        if cuda:
            self.hidden_states = [h.to(device) for h in self.hidden_states]
            self.cell_states = self.cell_states.to(device)
        
        return (self.hidden_states[0], self.cell_states)
    
    def forward(self, input, prev_state):
        if self.hidden_states is None:
            self.init_hidden(input.size(0), input.is_cuda, input.device)
        
        temp_states = self._process_temporal_states()
        
        # LSTM gates computation
        concat_conv = self.layers[-1](torch.cat([input, temp_states], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_dim, dim=1)
        
        i, f, o = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o)
        g = self.activation(cc_g)
        
        self.cell_states = f * self.cell_states + i * g
        outputs = o * self.activation(self.cell_states)
        
        self._update_hidden_buffer(outputs)
        return outputs, (outputs, self.cell_states)

class ConvTTGRUCell(BaseConvTTCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, 
                 activation=F.tanh, order=3, steps=3, ranks=8, dilation=1):
        super().__init__(input_size, input_dim, hidden_dim, kernel_size, bias, activation, dilation, order, steps, ranks)
        
        # Spatial processing layers for GRU gates
        self.layers_zr = nn.ModuleList()
        self.layers_h = nn.ModuleList()
        for l in range(order):
            in_channels = ranks if l < order - 1 else ranks + input_dim
            out_channels_zr = ranks if l < order - 1 else 2 * hidden_dim
            out_channels_h = ranks if l < order - 1 else hidden_dim
            self.layers_zr.append(self._get_conv2d(in_channels, out_channels_zr))
            self.layers_h.append(self._get_conv2d(in_channels, out_channels_h))
        
        self.reset_parameters()
    
    def _get_spatial_output(self, layer_idx, temp_states):
        return self.layers_zr[layer_idx](temp_states)
    
    def init_hidden(self, batch_size, cuda=True, device='cuda'):
        self.hidden_states = [torch.zeros(batch_size, self.hidden_dim, self.height, self.width) 
                             for _ in range(self.steps)]
        self.hidden_pointer = 0
        
        if cuda:
            self.hidden_states = [h.to(device) for h in self.hidden_states]
        
        return self.hidden_states[0]
    
    def forward(self, input, h_prev):
        if self.hidden_states is None:
            self.init_hidden(input.size(0), input.is_cuda, input.device)
        
        temp_states = self._process_temporal_states()
        
        # GRU gates computation
        zr_conv = self.layers_zr[-1](torch.cat([input, temp_states], dim=1))
        z, r = torch.split(torch.sigmoid(zr_conv), self.hidden_dim, dim=1)
        
        # Reset-modulated computation
        temp_states_reset = temp_states * r
        h_conv = self.layers_h[-1](torch.cat([input, temp_states_reset], dim=1))
        h_new = self.activation(h_conv)
        
        outputs = (1 - z) * h_new + z * self.hidden_states[self.hidden_pointer]
        self._update_hidden_buffer(outputs)
        return outputs

# ConvLSTM Cells
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
        if conv_type == 'tt':
            conv_kwargs.update(kwargs)
        
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

# ConvGRU Cells
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
        if conv_type == 'tt':
            conv_kwargs.update(kwargs)
        
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

# Main RNN classes
class ConvLSTM(BaseConvRNN):
    def __init__(self, cell_class=ConvLSTMCell, **kwargs):
        super().__init__(cell_class=cell_class, **kwargs)

    def forward(self, input, hidden_state):
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))
        
        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0].size(int(not self.batch_first)))

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(len(cur_layer_input)):
                h, c = self.cell_list[layer_idx](cur_layer_input[t], [h, c])
                output_inner.append(h)
            cur_layer_input = output_inner
            hidden_state[layer_idx] = (h, c)

        return torch.stack(output_inner, dim=int(self.batch_first)), hidden_state

    def get_init_states(self, batch_size, cuda=True, device='cuda'):
        return [self.cell_list[i].init_hidden(batch_size, cuda, device) for i in range(self.num_layers)]

class ConvGRU(BaseConvRNN):
    def __init__(self, cell_class=ConvGRUCell, **kwargs):
        super().__init__(cell_class=cell_class, **kwargs)

    def forward(self, input, hidden_state):
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))
        
        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0].size(0))

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(len(cur_layer_input)):
                h = self.cell_list[layer_idx](cur_layer_input[t], h)
                output_inner.append(h)
            cur_layer_input = output_inner
            hidden_state[layer_idx] = h

        return torch.stack(output_inner, dim=int(self.batch_first)), hidden_state

    def get_init_states(self, batch_size, cuda=True, device='cuda'):
        return [self.cell_list[i].init_hidden(batch_size, cuda, device) for i in range(self.num_layers)]

class AttentionRNN(nn.Module):
    def __init__(self, rnn_type="LSTM", **kwargs):
        super().__init__()
        
        rnn_class = ConvLSTM if rnn_type in ("LSTM", "ConvLSTM") else ConvGRU
        channel_rnn_class = nn.LSTM if rnn_type in ("LSTM", "ConvLSTM") else nn.GRU
        
        self.spatial_rnn = rnn_class(
            input_size=kwargs["input_size"], 
            input_dim=2, 
            hidden_dim=1, 
            **{k: v for k, v in kwargs.items() if k not in ["input_size", "input_dim", "hidden_dim", "bias"]},
            bias=False
        )
        self.channel_rnn = channel_rnn_class(
            input_size=kwargs['input_dim'], 
            hidden_size=kwargs['hidden_dim'], 
            bias=False, 
            batch_first=True
        )

    def forward(self, x, h=None):
        h_avg, h_max, h_spatial = h if h is not None else (None, None, None)

        # Channel attention
        avg_out, h_avg = self.channel_rnn(x.mean((3, 4)), h_avg)
        max_out, h_max = self.channel_rnn(x.amax((3, 4)), h_max)
        x = x * torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)

        # Spatial attention
        spatial_input = torch.cat([x.mean(dim=2, keepdim=True), x.max(dim=2, keepdim=True)[0]], dim=2)
        spatial_att, h_spatial = self.spatial_rnn(spatial_input, h_spatial)

        return x * torch.sigmoid(spatial_att), (h_avg, h_max, h_spatial)

# Factory functions for different variants
def SepConvLSTM(**kwargs):
    return ConvLSTM(conv_type='depthwise', **kwargs)

def SepConvGRU(**kwargs):
    return ConvGRU(conv_type='depthwise', **kwargs)

def TTConvLSTM(**kwargs):
    return ConvLSTM(ConvTTLSTMCell, **kwargs)

def TTConvGRU(**kwargs):
    return ConvGRU(ConvTTGRUCell, **kwargs)

def AttentionLSTM(**kwargs):
    return AttentionRNN(rnn_type="LSTM", **kwargs)

def AttentionGRU(**kwargs):
    return AttentionRNN(rnn_type="GRU", **kwargs)