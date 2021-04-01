import numpy as np
from tensorflow import keras

# y = 5x + 2
x = np.array([ -3.0, -2.0, -1.0, 0.0, 1.0,  2.0,  3.0], dtype=float)
y = np.array([-13.0, -8.0, -3.0, 2.0, 7.0, 12.0, 17.0], dtype=float)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x, y, epochs=500)

output = model.predict([10.0])

assert 51.999 < output[0][0] < 52.001
