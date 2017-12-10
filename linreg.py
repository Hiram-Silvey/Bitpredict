from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas

dataframe = pandas.read_csv('./data/200days.csv', sep=',', header=None)
dataset = np.flipud(dataframe.values)
sc = StandardScaler()

X_train = np.arange(1, 141)
X_train = X_train.reshape(-1, 1)
Y_train = dataset[:140,4]
Y_train = Y_train.reshape(-1, 1)
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train)

X_test = np.arange(141, 157)
X_test = X_test.reshape(-1, 1)
Y_test = dataset[140:,4]
Y_test = Y_test.reshape(-1, 1)
X_test = sc.fit_transform(X_test)
Y_test = sc.fit_transform(Y_test)

model = Sequential()
model.add(Dense(1, activation='linear', input_shape=(1,)))
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, Y_train, batch_size=1, verbose=1, epochs=10, shuffle=False)
plt.scatter(X_train, Y_train, color='black')
plt.plot(X_train, model.predict(X_train), color='blue', linewidth=3)
plt.show()