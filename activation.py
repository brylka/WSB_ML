import numpy as np
import pandas as pd
import plotly.express as px


np.set_printoptions(precision=6)

#ReLU
def max_relu(x):
    return max(x, 0)

for i in [-10, -5, 0, 5, 10]:
    print(max_relu(i))


data = np.random.randn(50)
data = sorted(data)
print(data)

# wygenerowanie danych do wykresu
max_relu_data = np.array([max_relu(x) for x in data])
print (max_relu_data)


df = pd.DataFrame({'data': data, 'max_relu_data': max_relu_data})
print (df.head())

px.line(df, x='data', y='max_relu_data', width=700, height=400, title='ReLU Function').show()


# Funkcja Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in [-5., -3., -1., 0., 1., 3., 5.]:
    print(sigmoid(i))

data = 3 * np.random.randn(50)
data = sorted(data)
print(data)

sigmoid_data = [sigmoid(x) for x in sorted(data)]
print (sigmoid_data)

df = pd.DataFrame({'data': data, 'sigmoid_data': sigmoid_data})
print(df.head())

#px.line(df, x='data', y='sigmoid_data', width=700, height=400, title='Sigmoid Function').show()




# Funkcja Tanh
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

data = 2 * np.random.randn(100)
data = sorted(data)
print (data)

tanh_data = [tanh(x) for x in sorted(data)]
print(tanh_data)

df = pd.DataFrame({'data': data, 'tanh_data': tanh_data})
print(df.head())

#px.line(df, x='data', y='tanh_data', width=700, height=400, title='Tanh Function').show()



# Funkcja Softmax
def softmax(x):
    e_x = np.exp(x)
    denominator = np.sum(e_x, axis=1)
    denominator = denominator[:, np.newaxis]
    return e_x / denominator

data = np.random.randn(4, 5)
print (data)

result = softmax(data)
print (result)

print (result.sum(axis=1))
