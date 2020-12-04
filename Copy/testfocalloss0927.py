"""
date：0927
测试编写focal loss

note：1006 接着之前的内容看一看
"""


import tensorflow as tf
from keras import backend as K

def binary_focal_loss(gamma=2,alpha=0.25):
    alpha=tf.constant(alpha,dtype=tf.float32)
    gamma=tf.constant(gamma,dtype=tf.float32)

    def binary_focal_loss_fixed(y_true,y_pred):
        y_true=tf.cast(y_true,tf.float32)#数据类型转换
        alpha_t=y_true*alpha+(K.ones_like(y_true)-y_true)*(1-alpha)

        p_t=y_true*y_pred+(K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred)+K.epsilon()
        focal_loss=-alpha_t*K.pow((K.ones_like(y_true)-p_t),gamma)*K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed





def binary_focal_loss_fixed(y_true,y_pred):
    alpha=0.25
    gamma=2
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    y_true=tf.cast(y_true,tf.float32)#数据类型转换
    alpha_t=y_true*alpha+(K.ones_like(y_true)-y_true)*(1-alpha)

    p_t=y_true*y_pred+(K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred)+K.epsilon()
    focal_loss=-alpha_t*K.pow((K.ones_like(y_true)-p_t),gamma)*K.log(p_t)
    return K.mean(focal_loss)

import keras


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # 找出存在目标的先验框
        indices_for_object        = backend.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = backend.gather_nd(labels, indices_for_object)
        classification_for_object = backend.gather_nd(classification, indices_for_object)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_object = keras.backend.ones_like(labels_for_object) * alpha
        alpha_factor_for_object = backend.where(keras.backend.equal(labels_for_object, 1), alpha_factor_for_object, 1 - alpha_factor_for_object)
        focal_weight_for_object = backend.where(keras.backend.equal(labels_for_object, 1), 1 - classification_for_object, classification_for_object)
        focal_weight_for_object = alpha_factor_for_object * focal_weight_for_object ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_object = focal_weight_for_object * keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = backend.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = backend.gather_nd(labels, indices_for_back)
        classification_for_back = backend.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_back = keras.backend.ones_like(labels_for_back) * (1 - alpha)
        focal_weight_for_back = classification_for_back
        focal_weight_for_back = alpha_factor_for_back * focal_weight_for_back ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_back = focal_weight_for_back * keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object)
        cls_loss_for_back = keras.backend.sum(cls_loss_for_back)

        # 总的loss
        loss = (cls_loss_for_object + cls_loss_for_back)/normalizer

        return loss
    return _focal