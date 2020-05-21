import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense


model=Sequential()
model.add(Dense(64,activation='relu',input_dim=100))

#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# loss=keras.losses.categorical_crossentropy
# optimizer=keras.optimizers.SGD(lr=0.01,)
#
# #model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.op)
#
# model.fit()
#
# model.train_on_batch()
#
# loss_and_metrics=model.evaluate()

model.add(layers.Dense(32,activation='relu'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_crossentropy])
