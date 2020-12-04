"""
20201117 测试计算FLOPs的模板代码，准备在这之上进行修改

运算结果：
WARNING:tensorflow:From D:\software\Anaconda\envs\python36\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From D:\software\Anaconda\envs\python36\lib\site-packages\tensorflow\python\profiler\internal\flops_registry.py:142: tensor_shape_from_node_def_name (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.compat.v1.graph_util.remove_training_nodes
5 ops no flops stats due to incomplete shapes.
Incomplete shape.
Incomplete shape.
Parsing Inputs...
total_params 41 total_flops 33


"""

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
     def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

     def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)



model = MyModel()
inputs = tf.keras.layers.Input(shape=(3,3,3))
outputs = model(inputs)
model = tf.keras.Model(inputs, outputs)

def num_params_flops(model,readable_format=False):

    """Return number of parameters and flops."""
    nparams = np.sum(
    [np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    flops = tf.compat.v1.profiler.profile(
      tf.compat.v1.get_default_graph(), options=options).total_float_ops
    # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
    flops = flops // 2
    if readable_format:
        nparams = float(nparams) * 1e-6
        flops = float(flops) * 1e-9
    return nparams, flops

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

nparams, flops = num_params_flops(model)
print('total_params {}'.format(nparams), 'total_flops {}'.format(flops))

flops = num_flops()
print('total_flops {}'.format(flops))
