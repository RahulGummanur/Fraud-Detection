# # Fraud Detection

# Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "MiniSom"])

# Importing the Dataset

data_set = pd.read_csv('Credit_Card_Applications.csv')
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# # Training the SOM

from minisom import MiniSom
som = MiniSom(10,10,input_len = 15,sigma = 1.0,learning_rate= 0.5)
som.random_weights_init(X)
print(som.train_random(data = X,num_iteration=100))

# Visualing the Data

from pylab import bone, pcolor, colorbar, show, plot
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
  w = som.winner(x)
  plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
print(show())

# # Finding the frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,2)],mappings[(6,1)],mappings[(8,1)]),axis = 0)
frauds = sc.inverse_transform(frauds)

# Printing the Fraud Clients

print('Fraud Client IDs')
for i in frauds[:,0]:
  print(int(i))

# # Unsupervised to supervised Neural Network

# Creating a matrix of features

customers = data_set.iloc[:,1:].values

# Creating a Dependant Variable for Supervised DL Model

is_fraud = np.zeros(len(data_set))
for i in range(len(data_set)):
  if data_set.iloc[i,0] in frauds :
    is_fraud[i] = 1

# Feature scaling for ANN

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
customers =  sc.fit_transform(customers)

# # Building The ANN

import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2,activation='relu'))
print(ann.add(tf.keras.layers.Dense(units =1,activation = 'sigmoid')))

# Compiling the ANN

print(ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy']))

# Training the ANN

print(ann.fit(customers,is_fraud,batch_size = 32,epochs = 5))

# Predicting the Output

y_pred = ann.predict(customers)
y_pred = np.concatenate((data_set.iloc[:,0:1].values,y_pred),axis =1)
y_pred = y_pred[y_pred[:,1].argsort()]
print(y_pred)
