#housepriceprediction
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([0, 1, 2, 4, 6, 8, 10])
ys = np.array([50, 100, 150, 250, 350, 450, 550])
model.fit(xs, ys, epochs=100)
print(model.predict([7.0])) #should be 50+(7*50)=400
