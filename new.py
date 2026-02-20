import numpy as np

l: int = 2  # total layer count

x_min: float = 0.0
x_max: float = 1.0

# layer specific
omega_l: np.ndarray = np.linspace(x_min, x_max, l)  # Equidistant grid in layer l
h_l: float = 2 ** (-l)
I_l: list = list(range(2**l))  # Gridpoint indices   i \in {0,1,..., 2**l}


def x_li(l, i):
    return i * 2 ** (-l)  # Value of function at gridpoints


i = 3

print(l)
print(i)
print(h_l)
print(I_l)
print(x_li(2, 3))
