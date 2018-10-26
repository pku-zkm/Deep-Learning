import tensorflow as tf
import tensorflow.train
from tensorflow import keras
import numpy as np

model=keras.Sequential()
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss='categorical_crossentropy',matrics=['accuracy'])

data=np.random.random((100000,32))
labels=np.random.random((100000,10))
model.fit(data,labels,epochs=10,batch_size=32)