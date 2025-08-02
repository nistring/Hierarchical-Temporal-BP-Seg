from .convrnn import ConvLSTM, ConvGRU

def SepConvLSTM(**kwargs):
    return ConvLSTM(conv_type='depthwise', **kwargs)

def SepConvGRU(**kwargs):
    return ConvGRU(conv_type='depthwise', **kwargs)