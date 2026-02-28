import matplotlib.pyplot as plt
import numpy as np

l: int = 4  # total layer count

x_min: float = 0.0
x_max: float = 1.0

# layer specific
omega_l: np.ndarray = np.linspace(x_min, x_max, 2**l + 1)  # Equidistant grid in layer l
h_l: float = 2 ** (-l)
I_l: list = list(range(2**l))  # Gridpoint indices   i \in {0,1,..., 2**l}


def x_li(l: int, i: int) -> float:
    """x coordinate of gridpoint at index i on level l"""
    return i * 2 ** (-l)  # Value of function at gridpoints
    # or return i* h_l


def phi(x: float) -> float:
    """
    Hat funciton centered at 0

    Parameters
        x: float x coordinate
    Returns
        float value of the hat function at x
    """
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0


i = 3

def phi_li(x: float, l: int, i: int, h_l: float) -> float:
    """
    Base function for gridpoint x_li(l, i) on level l
    with phi_li(x_li) = 1 on [x_li - h_l, x_li + h_l]. 1 in the center and 0 at the borders

    Parameters
        x: float x coordinate
        l: int level
        i: int index of grid point
        h_l: float grid spacing on level l
    Returns
        float value of the base function at x
    """
    return phi((x - x_li(l, i)) / h_l)


assert phi_li(0.75, 2, 3, h_l) == 1.0
assert phi_li(1, 2, 3, h_l) == 0.0

def f(x):
    return 2.5*np.sin(np.pi*x)*x**2

def u()

# for k in range(l):
#     n = 2**k
#     x_coords = np.array([x_li(k, i) for i in range(n)])
#     print("Level: ", k)
#     print("Number of grid points: ", n)
#     print("Coords: ", x_coords)
#
#     # Evaluate function at all gridpoints in level l
#     # y_l = [phi_li(x_coords, k,) for i in range()]


def f(x: np.float64 | np.ndarray | float) -> np.float64 | np.ndarray:
    return 2.5 * np.sin(np.pi * x) * x**2
