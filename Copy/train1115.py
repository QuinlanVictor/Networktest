"""
Retrain the YOLO model for your own dataset.、

note 1103 测试双尺度的训练
     1111 测试加入CBAM的网络结构，把tiny_yolo的部分删除了
          测试成功，能够正常开始训练

          -2 测试一下SE模块
          测试成功，能够正常开始训练

     1114 测试一下resnet50为骨干网络
          倒是能跑通，就是这骨干网络是不是太大了，我还得再看看

     1115 测试余弦退火调整学习率
          倒是能够训练，就是不知道训练效果如何

     1117 测试一下tensorboard
          打开cmd，进入工作目录下，输入 ‘tensorboard --logdir=logs’

          测试计算一下FLOPs
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from yolo3.model1111_2 import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from yolo3.utils1115 import WarmUpCosineDecayScheduler

import json
import tensorflow as tf

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def _main():
    
    annotation_path = 'traintest.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors6.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (512,512) # multiple of 32, hw


    model = create_model(input_shape, anchors, num_classes,
        freeze_body=True) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir,histogram_freq=1,write_grads=True)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    stage1_epochs = 1
    stage2_epochs = 1


    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:


        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        warmup_epoch = int(stage1_epochs * 0.2)
        total_steps = int(stage1_epochs * num_train / batch_size)
        warmup_steps = int(warmup_epoch * num_train / batch_size)
        learning_rate_base = 1e-3
        reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                               total_steps=total_steps,
                                               warmup_learning_rate=1e-4,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=num_train,
                                               min_learn_rate=1e-6
                                               )

        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        history1 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=stage1_epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        
        np.savez('trainHistory1.npz', history = history1.history)
        with open('trainHistory1.json', 'w') as f:
            json.dump(history1.history, f, cls = MyEncoder )#编码json文件
        print('Setp1 done! Save history to trainHistory1.json successfully!')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        print('Unfreeze all of the layers.')

        batch_size = 1 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        warmup_epoch = int(stage2_epochs * 0.2)
        total_steps = int(stage2_epochs * num_train / batch_size)
        warmup_steps = int(warmup_epoch * num_train / batch_size)
        learning_rate_base = 1e-4
        reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                               total_steps=total_steps,
                                               warmup_learning_rate=1e-5,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=num_train,
                                               min_learn_rate=1e-6
                                               )

        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change

        history2 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=stage1_epochs + stage2_epochs,
            initial_epoch=stage1_epochs,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

        np.savez('trainHistory2.npz', history = history2.history)
        with open('trainHistory2.json', 'w') as f:
            json.dump(history2.history, f, cls = MyEncoder )
        print('Setp2 done! Save history to trainHistory2.json successfully!')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def num_flops(readable_format=False):
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    flops = tf.compat.v1.profiler.profile(
        tf.compat.v1.get_default_graph(), options=options).total_float_ops
    # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
    flops = flops // 2
    if readable_format:
        flops = float(flops) * 1e-9
    return flops


def create_model(input_shape, anchors, num_classes, freeze_body=True):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:8,1:4}[l], w//{0:8,1:4}[l], num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = yolo_body(image_input, num_anchors//2, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if freeze_body:
        num = len(model_body.layers) - 2

        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    print('The model has {} layers .'.format(len(model_body.layers)))
    #plot_model(model,to_file='model_data/model1114.png',show_shapes=True,show_layer_names=True) #储存网络结构
    model.summary()#打印网络
    flops = num_flops()
    print('total_flops {}'.format(flops))


    return model



def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
