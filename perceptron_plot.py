import matplotlib.pyplot as plt
import numpy as np

def perceptron(x):
    # Standard step function activation
    return np.where(x > 0, 1, 0) 

w = -2
b = 3
z = np.linspace(-10, 10, 1000)
x = w * z + b

y = perceptron(x)

plt.plot(z, y) # Changed x to z to see the impact of w and b
plt.title(f'Perceptron (Step) with w={w}, b={b}')
plt.xlabel('Input (z)')
plt.ylabel('Output (0 or 1)')
plt.grid(True, linestyle='--')
plt.show()