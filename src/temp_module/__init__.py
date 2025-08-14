from .base import BaseConvRNN, BaseConvLSTM
from .cell import ConvLSTMCell, ConvGRUCell, ConvRNNCell, MinConvLSTMCell, MinConvGRUCell, MinConvExpLSTMCell, ReducedConvLSTMCell

class ConvLSTM(BaseConvLSTM):
    def __init__(self, **kwargs):
        super().__init__(cell_class=ConvLSTMCell, **kwargs)

class ReducedConvLSTM(BaseConvLSTM):
    def __init__(self, **kwargs):
        super().__init__(cell_class=ReducedConvLSTMCell, **kwargs)

class ConvGRU(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=ConvGRUCell, **kwargs)

class ConvRNN(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=ConvRNNCell, **kwargs)

class MinConvLSTM(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=MinConvLSTMCell, **kwargs)

class MinConvGRU(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=MinConvGRUCell, **kwargs)

class MinConvExpLSTM(BaseConvRNN):
    def __init__(self, **kwargs):
        super().__init__(cell_class=MinConvExpLSTMCell, **kwargs)