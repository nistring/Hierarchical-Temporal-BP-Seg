import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=padding, groups=in_channels, bias=False, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

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