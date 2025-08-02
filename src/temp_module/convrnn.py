# Adopted from https://github.com/aserdega/convlstmgru/blob/master/convlstm.py and https://github.com/aserdega/convlstmgru/blob/master/convgru.py
import torch
from .base import BaseConvRNN
from .cell import ConvLSTMCell, ConvGRUCell

class ConvLSTM(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=ConvLSTMCell, **kwargs)

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
    def __init__(self, **kwargs):
        super().__init__(cell_class=ConvGRUCell, **kwargs)

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