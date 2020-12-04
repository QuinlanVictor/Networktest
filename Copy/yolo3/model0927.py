"""
20200927
测试设计出来的网络结构，基于yolov3对于肺结节检测这一任务进行改进，减少darknet53的单元数量，替换骨干网络的网络模块，
替换激活函数，在骨干网络后加入spp，对于检测网络进行修改，将三个尺度的输出变为一个尺度是输出，同时对于损失函数做修改，
替换中心坐标xy和宽高wh的损失函数，变为ciou_loss

在研究中继续改进，可能会有所改变。

note:0929 初步搭建网络完成，按照ppt上设计的网络结构，需要去考虑后续怎么修改计算，只计算一个输出,看看如何修改
     1003 进行多维数组计算的研究，搞明白shape的变化，以及用不用修改，把想法注释在代码上
     1005 检查一下是否还有什么问题，然后进行训练测试一下
     1006 因为昨晚网络没有训练完，所以预计是今天进行一下测试，准备新建一个项目来测试新的网络结构。
          然后看看代码，修改一下之前没有注意的地方
     1007 很不幸的是，昨天下午折腾了一下午，怎么修改都没办法通过debug，好像还是在model_loss那里出现了问题，说明还是尺度或者shape哪里
     没有设置好，感觉需要再重新过一遍代码。不过从修改中得到的经验来看，改为两尺度输出应该还是挺容易的，毕竟tiny的输出就是两尺度，还是简
     单一些的。
"""
from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

tf.python.control_flow_ops=tf
"""网络结构部分"""
"""Mish激活函数"""
class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

"""conv"""
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):

    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

"""实现CBM模块"""
def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

"""实现CBL模块"""
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

"""cspblock"""
def cspblock_body(x, num_filters, num_blocks, all_narrow=True):
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)

    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)

    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)

    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])

    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    route = Concatenate()([postconv, shortconv])

    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

"""网络主体结构"""
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = cspblock_body(x, 64, 1, False)
    x = cspblock_body(x, 128, 2)
    x = cspblock_body(x, 256, 3)
    feat1 = x
    x = cspblock_body(x, 512, 3)
    feat2 = x
    x = cspblock_body(x, 1024, 2)
    feat3 = x
    return feat1,feat2,feat3

"""三次卷积的输出层"""
def outlayer_threeconvs(x, num_filters):
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Mish(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (1,1))(x)
    return x

"""SPP模块"""
def sppmodule(x,num_filters):
    x = compose(
        DarknetConv2D_BN_Mish(num_filters,(1,1)),
        DarknetConv2D_BN_Mish(num_filters*2,(3,3)),
        DarknetConv2D_BN_Mish(num_filters,(1,1))
    )(x)

    maxpool1 = MaxPooling2D(pool_size=(13,13),strides=(1,1),padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(8,8),strides=(1,1),padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(5,5),strides=(1,1),padding='same')(x)

    x=Concatenate()([maxpool1,maxpool2,maxpool3,x])

    x = compose(
        DarknetConv2D_BN_Mish(num_filters, (1, 1)),
        DarknetConv2D_BN_Mish(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Mish(num_filters, (1, 1))
    )(x)

    return x

"""yolo主体"""
def yolo_body(inputs,num_anchors,num_classes):
    feat1, feat2, feat3=darknet_body(inputs)

    #32倍下采样的输出
    x1=sppmodule(feat3,512)
    x1=outlayer_threeconvs(x1,512)

    x1=DarknetConv2D_BN_Mish(512,(1,1))(x1)
    x1=UpSampling2D(2)(x1)

    #16倍下采样的输出
    x2=Concatenate()([x1,feat2])
    x2=outlayer_threeconvs(x2,256)

    x2=DarknetConv2D_BN_Mish(256,(1,1))(x2)
    x2=UpSampling2D(2)(x2)

    #8倍下采样的输出,只保留这一个尺度的输出
    x3=Concatenate()([x2,feat1])
    x3=outlayer_threeconvs(x3,128)

    x3=DarknetConv2D_BN_Mish(128,(1,1))(x3)
    x3=DarknetConv2D(num_anchors*(num_classes+5),(1,1))(x3)

    return Model(inputs,x3)


"""loss及相关计算部分"""

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [0,1,2] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

#计算y_true，注意维度，‘……’操作符只操作最前或最后一维
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # 这里要不要除以3呢，这里使用的就是3个prior，3个先验框
    #好像可以忽略layer这个参数了，就一层没有必要再设置
    anchor_mask = [0,1,2]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//8]
    y_true = np.zeros((m,grid_shapes[0],grid_shapes[1],len(anchor_mask),5+num_classes),
        dtype='float32')
    #此时y_true为全0列表，(16,52,52,3,6),注意一下数组维度的修改
    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        #wh将标记为True的框筛选出来
        wh = boxes_wh[b, valid_mask[b]]  #到这一步的时候shape是不一定的，是以(x,2)的shape，x为多少取决于true的范围，这块应该是不用修改的
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)#倒数第二个添加一位，(7,2)变为(7,1,2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        #标注框BOX和anchor box的iou值计算
        #intersect_area为(7,9)，box_area为(7,1)，anchor_area为(1,9)——iou的shape为(7,9)
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)#(7,1)+(1,9)=(7,9)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            if n in anchor_mask:
                i = np.floor(true_boxes[b,t,0]*grid_shapes[1]).astype('int32')
                j = np.floor(true_boxes[b,t,1]*grid_shapes[0]).astype('int32')
                k = anchor_mask.index(n)
                c = true_boxes[b,t, 4].astype('int32')
                y_true[b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                y_true[b, j, i, k, 4] = 1
                y_true[b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)



    return iou

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss


    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [0,1,2]
    input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 8, K.dtype(y_true))
    grid_shapes = K.cast(K.shape(yolo_outputs)[1:3], K.dtype(y_true))
    loss = 0
    m = K.shape(yolo_outputs)[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss

