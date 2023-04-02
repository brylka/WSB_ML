import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.random import set_seed

set_seed(0)

# komponenty sieci
model = Sequential()
#                       linear, relu, sigmoid, tanh, softmax
model.add(Dense(4, input_shape=[1], activation="linear"))
model.add(Dense(2, activation="linear"))
model.add(Dense(1))

#kompilacja modelu
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
df = pd.read_csv("https://vesemir.wiedzmin.net/extras/f-c.csv", usecols=[1,2])

print(df.head())

plt.scatter(df.F, df.C)
#plt.show()

# uczenie sieci
result = model.fit(df.F, df.C, epochs=1500, verbose=0)

print(result.history.keys())

df1 = pd.DataFrame(result.history)
print(df1.head(500))

#df1.plot()
#plt.show()

y_pred = model.predict(df.F)

plt.scatter(df.F, df.C)

plt.plot(df.F, y_pred, c='r')

plt.show()
