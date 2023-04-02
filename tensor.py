import numpy as np

# Skalar
scalar = 3
print(scalar)
print(type(scalar))

scalar = 3.0
print(scalar)
print(type(scalar))

# Wektor
vector = np.array([2, 4, -6, 5])
print(vector)
print(type(vector))
print(f'Rozmiar wektora: {vector.shape}')
print(f'Typ danych wektora: {vector.dtype}')
print(f'Rząd: {vector.ndim}')
print(f'Długość: {len(vector)}')

vector = np.array([2, 4, -6, 5], dtype='float')
print(vector)
print(f'Typ danych wektora: {vector.dtype}')

# Macierz
array = np.array([[2, 6, 3],
                  [5, -3, 4]])
print(array)
print(type(array))
print(f'Rozmiar macierzy: {array.shape}')
print(f'Typ danych macierzy: {array.dtype}')
print(f'Rząd: {array.ndim}')
print(f'Długosc: {len(array)}')




array = np.array([[2, 6, 3],
                  [5, 3, 4],
                  [4, 2, 1]], dtype='float')
print(array)
print(f'Rozmiar macierzy: {array.shape}')
print(f'Typ danych macierzy: {array.dtype}')
print(f'Rząd: {array.ndim}')
print(f'Długosc: {len(array)}')


# Tensor
tensor = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[7, 8, 9],
     [3, 7, 3]]
])

print(tensor)
print(type(tensor))
print(f'Rozmiar macierzy: {tensor.shape}')
print(f'Typ danych macierzy: {tensor.dtype}')
print(f'Rząd: {tensor.ndim}')
print(f'Długosc: {len(tensor)}')




tensor = np.array([
    [[1, 2, 3],
     [4, 5, 6]],
    [[7, 8, 9],
     [3, 7, 3]],
    [[2, 3, 5],
     [7, 2, 5]]
])

print(tensor)
print(type(tensor))
print(f'Rozmiar macierzy: {tensor.shape}')
print(f'Typ danych macierzy: {tensor.dtype}')
print(f'Rząd: {tensor.ndim}')
print(f'Długosc: {len(tensor)}')



tensor = np.array([
    [[1, 2, 3, 4],
     [4, 5, 6, 4],
     [4, 2, 5, 2]],
    [[7, 8, 9, 8],
     [3, 7, 3, 9],
     [5, 2, 4, 3]],
    [[2, 3, 5, 4],
     [7, 2, 5, 1],
     [8, 2, 7, 2]],
    [[2, 3, 5, 7],
     [7, 2, 5, 9],
     [8, 2, 7, 0]],
    [[2, 3, 5, 7],
     [7, 2, 5, 9],
     [8, 2, 7, 0]]
])

print(tensor)
print(type(tensor))
print(f'Rozmiar macierzy: {tensor.shape}')
print(f'Typ danych macierzy: {tensor.dtype}')
print(f'Rząd: {tensor.ndim}')
print(f'Długosc: {len(tensor)}')



print(tensor[0:2][0])
