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


def f(x: np.float64 | np.ndarray | float) -> np.float64 | np.ndarray:
    return 2.5 * np.sin(np.pi * x) * x**2


def u(x: float) -> float:
    plt.title("Hierarchical approximation")
    plt.xlim(0, 1)

    I_l = [i for i in range(2**l)]  # whole domain omega_l
    alpha = np.zeros((l, len(I_l)))
    y_pred = np.zeros_like(alpha)

    # for all levels
    for k in range(1, l):
        h_k = 2 ** (-k)
        I_k = [i for i in range(1, 2**k) if i % 2 == 1]

        preds = np.zeros(len(I_l))

        # For all base functions on level
        alphas_k = np.zeros(len(I_k))
        for i in I_k:
            alphas_k[i] = f(x_li(l, i)) - phi_li(x_li(k, i), k, i, h_k)

        # compute pred for every point on the finest grid
        for i in I_l:
            alpha[k, i] = f(x_li(l, i)) - preds[i]

    return y_pred[-1, :]


def u2(x):
    plt.title("Hierarchical approximation")
    plt.xlim(0, 1)
    I_l = [i for i in range(2**l)]

    for k in range(1, l + 1):
        index_of_base_function_centers = [i for i in range(2**k) if i % 2 == 1]
        num_base_functions_on_k = len(index_of_base_function_centers)
        x_coords_centers = [x_li(k, i) for i in index_of_base_function_centers]
        h_k = 2 ** (-k)

        print(
            f"k: {k} {'=' * 80}\n\t num_base_functions: {num_base_functions_on_k}, idx_of_base_functions: {index_of_base_function_centers},\n\t coords: {x_coords_centers}"
        )

        # PLOT W_k
        y_coords = np.zeros(1000)
        x_coords = np.linspace(0, 1, 1000)
        for j in range(len(x_coords)):
            for i in index_of_base_function_centers:
                y_coords[j] += phi_li(x_coords[j], k, i, h_k)

        plt.plot(x_coords, y_coords, label=rf"$W_{k}$")


result = u2(0.5)
x_coords = np.linspace(x_min, x_max, 100)

y_fx = f(x_coords)

# plt.plot(x_coords, y_fx, label="f(x)", color="r")
plt.legend()
plt.show()
