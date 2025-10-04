import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DepthwiseSeparableConv2d, PointwiseConv2d

class BaseConvRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, activation=F.tanh, dilation=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = ((kernel_size[0] - 1) * dilation) // 2, ((kernel_size[1] - 1) * dilation) // 2
        self.bias = bias
        self.activation = activation

    def _get_conv_layer(self, in_channels, out_channels, conv_type='standard', bias=True):
        conv_classes = {
            'standard': nn.Conv2d,
            'depthwise': DepthwiseSeparableConv2d,
            'pointwise': PointwiseConv2d,
        }
        conv_class = conv_classes.get(conv_type, nn.Conv2d)

        conv_kwargs = {
            'kernel_size': self.kernel_size, 
            'padding': self.padding, 
            'bias': bias, 
            'dilation': self.dilation,
        }
        return conv_class(in_channels, out_channels, **conv_kwargs)

    def init_hidden(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        state = torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device)
        return state

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    module.bias.data.zero_()

class BaseConvRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, activation=F.tanh, dilation=1, conv_type='standard', cell_class=None, **cell_kwargs):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        
        # Extend parameters for multilayer
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation = self._extend_for_multilayer(activation, num_layers)
        dilation = self._extend_for_multilayer(dilation, num_layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create cell list
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i-1]
            cell_list.append(cell_class(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim[i],
                kernel_size=kernel_size[i],
                bias=bias,
                activation=activation[i],
                dilation=dilation[i],
                conv_type=conv_type,
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

    def forward(self, input, hidden_state):
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))
        
        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0])

        for layer_idx in range(self.num_layers):
            output_inner = []
            h = hidden_state[layer_idx]
            for t in range(len(cur_layer_input)):
                h = self.cell_list[layer_idx](cur_layer_input[t], h)
                output_inner.append(h)
            hidden_state[layer_idx] = h
            
            cur_layer_input = output_inner

        return torch.stack(output_inner, dim=int(self.batch_first)), hidden_state

    def get_init_states(self, input_tensor):
        return [self.cell_list[i].init_hidden(input_tensor) for i in range(self.num_layers)]

class BaseConvLSTM(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, hidden_state):
        cur_layer_input = torch.unbind(input, dim=int(self.batch_first))
        
        if hidden_state is None:
            hidden_state = self.get_init_states(cur_layer_input[0])

        for layer_idx in range(self.num_layers):
            output_inner = []
            h, c = hidden_state[layer_idx][0], hidden_state[layer_idx][1]
            for t in range(len(cur_layer_input)):
                h, c = self.cell_list[layer_idx](cur_layer_input[t], [h, c])
                output_inner.append(h)
            hidden_state[layer_idx] = torch.stack([h, c])
            
            cur_layer_input = output_inner

        return torch.stack(output_inner, dim=int(self.batch_first)), hidden_state