import numpy as np

# num layers
l: int = 3
# list of grid spacing per level
h: list = [2 ** (-k) for k in range(l)]
# h_l: float = 2 ** (-l) grid spacing of the last level.
# grid point coordinates (x values)

# x_li

x_ki: np.ndarray = np.array([[i * h[k] for i in range(2**l + 1)] for k in range(l)])

num_points_per_layer = [2**k for k in range(l)]

# list of odd integers from 1 to 2^l - 1
I_l: list = [i for i in range(1, 2**l) if i % 2 == 1]


print("=" * 100)
print("Hut interpolation parameters:")
print(f"num layers: {l}")
print(f"grid spacing per level: {h}")
print(f"grid point coordinates: {x_ki}")
print(f"Num Grid points per layer: {num_points_per_layer}")
print(f"Basis function center indices: {I_l}")
print("=" * 100)


def hat_function(x: np.float64):
    """
    hat function for interpolation. To be used as a base function

    x: point to evaluate the hat function. This x needs to be tranformed to be centered at 0 and for h_l=1 already"""
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0


def base_function(x: float, i: int):
    """
    Base function for interpolation

    x: point to evaluate the base function
    i: index of grid point
    h_l: grid spacing
    """

    # x - i*h_l -> shifts the point back to be centered at 0
    # / h_l -> scales the gridspacing back to 1

    return hat_function((x - i * h_l) / (h_l))


# Test base function

print(base_function(x=0.75, i=3))


def u(x: float):
    """
    approximation of the function to be interpolated.

    x: point to evaluate the approximation at
    """

    for k in range(1, l + 1):
        for i in I_k:
            alpha_ki * base_function(x, k, i)
