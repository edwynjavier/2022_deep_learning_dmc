#Ejercicio: Creando una red neuronal usando TensorFlow y Keras

Basado en el siguiente código crearemos un notebook en el 
que construiremos paso a paso una red neuronal usando Tensorflow y Keras

#1: Invocar librerias

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets

#2: Obtener datos
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

#3: Definir datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train

#3: Crear un modelo vacio

# First, we initialize the model.
model = tf.keras.models.Sequential()

#4: Agregar capas al modelo
# First our input layer. For this layer, and this layer only, we need to specify the size of our input. For our dataset this means the amount of columns in our X.
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(X.shape[1],)))
# Now some hidden layers
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
# Finally, our output layer. Since we have 3 possible classes, we need 3 output neurons. 
# For a regression problem, we would have only 1. For an image creation network, we would have as many pixels as the image we wanted to create!
model.add(tf.keras.layers.Dense(3))
# A final layer with several output neurons gives us logits as results. 
#We can do a final pass with a Softmax layer to turn them into percentages.
model.add(tf.keras.layers.Softmax())

#5: Crear un optimizador
optim = tf.keras.optimizers.Adam(learning_rate=0.01)

#6: Compilar el modelo

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=optim,metrics=['accuracy'])

#7: Entrenar la red
history = model.fit(X_train.values, y_train.values,
        validation_data=(X_test.values, y_test.values),
        epochs = 20,
        batch_size=32)

#8: Visualizar el desempeño
plt.plot(history.history['loss'], label='Sparse Categorical Crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Sparse Categorical Crossentropy (validation data)')
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.legend(loc="upper left")
plt.show()

#9: Realizar predicciones (usando el dataset de prueba)
# Let's get the prediction for the first flower in the test set
model.predict(X_test[:1])

#10: Ver el desempeño
predictions = model.predict(X_test)
for idx, prediction in enumerate(predictions):
    print('We predict: '+str(np.argmax(prediction))+'. Real Species was: '+str(y_test.iloc[idx]))