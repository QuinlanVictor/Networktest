"""

注意力机制，对于SAM的测试代码，对于注意力模块进行探究
date：1003

note：1005 还是没有什么想法
      1006 借鉴一下

"""
from keras import backend as K
from keras.layers import Concatenate, Add, Multiply
from keras.layers import Conv2D
from keras.activations import sigmoid

def attention(inputs):
    input_channels = int(inputs.shape[-1])

    x = Conv2D



def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x