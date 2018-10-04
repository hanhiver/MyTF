import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import SGD

def main():
     X = np.zeros((4, 2), dtype='uint8')
     y = np.zeros(4, dtype='uint8')
     X[0] = [0, 0]
     y[0] = 0
     X[1] = [0, 1]
     y[1] = 1
     X[2] = [1, 0]
     y[2] = 1
     X[3] = [1, 1]
     y[3] = 0

     model = Sequential()
     model.add(Dense(2, input_dim=2))
     model.add(Activation('relu'))
     model.add(Dense(1))
     model.add(Activation('sigmoid'))

     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

     model.compile(loss='mse', optimizer=sgd)

     history = model.fit(X, y, epochs=1000, batch_size=4, verbose=2)

     print(model.predict(X))

if __name__=="__main__":
     main()