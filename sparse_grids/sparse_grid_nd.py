from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class SparseGrid2d:
    def __init__(self, l: int):
        self.l: int = l  # Total layer count

        self.levels = list(range(1, self.l + 1))
        # Grid spacing per level
        self.h = np.array(
            [2 ** (-k) for k in range(self.l + 1)]
        )  # get grid spacing for levels 0 to l, we will only use levels 1 to l, but this way we have the correct index for h[k]
        # Base function values.
        self.base_function_values = np.zeros((self.l + 1, 2**self.l + 1, 2**self.l + 1))
        self.base_function_values_1d = np.zeros((self.l + 1, 2**self.l + 1))

        # X points on finest grid
        self.x_l = np.linspace(0, 1, 2**self.l + 1)

    def x_li(self, l: int, i: int) -> float:
        """x coordinate of gridpoint at index i on level l"""
        return i * 2 ** (-l)

    def phi(self, x: float) -> float:
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

    def phi_li(self, x: float | np.float64, k: int, i: int) -> float:
        """
        Base function for gridpoint x_li(l, i) on level l
        with phi_li(x_li) = 1 on [x_li - h_l, x_li + h_l]. 1 in the center and 0 at the borders

        Parameters
            x: (float) x coordinate
            k: (int) level
            i: (int) index of the grid point of the base function (on level k)
        Returns
            float value of the base function at x
        """
        return self.phi((x - self.x_li(k, i)) / self.h[k])

    def indices_of_funcs_on_k(self, k: int) -> list[int]:
        """
        Indexset. I_k = {i \in {1,..., 2**k-1} | i is odd}

        Calculates the indexes of the base functions on level k.
        On each level the base functions are centered on the uneven indexes
        and have support on [x_li(k, i) - h[k], x_li(k, i) + h[k]]
        """
        return [i for i in range(1, 2**k) if i % 2 == 1]

    def calculate_base_functions(self) -> None:
        """
        Builds the base functions for all levels and saves the values at the resolution of the finest level.
        """
        for k in self.levels:
            I_k = self.indices_of_funcs_on_k(k)

            for base_func_idx in I_k:
                for i in range(len(self.x_l)):
                    self.base_function_values_1d[k, i] += self.phi_li(
                        float(self.x_l[i]), k, base_func_idx
                    )

            # Tenorrproduct w_k1,w_k2 from w_k1 and w_k2
            self.base_function_values[k] = np.tensordot(
                self.base_function_values_1d[k], self.base_function_values_1d[k], axes=0
            )

    def visualize_base_functions_values_2d(self) -> None:
        """
        Plots the base functions on different levels for 2d
        """

        X, Y = np.meshgrid(self.x_l, self.x_l)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, self.base_function_values[self.l], alpha=0.8)
        fig.colorbar(surf)
        plt.title(f"Base functions of levels 1-{self.l}")
        plt.show()

    def visualize_base_function_values_1d(self) -> None:
        """
        Plots the base functions on different levels.
        """
        plt.title(f"Base functions of levels 1-{self.l}")
        for k in self.levels:
            plt.plot(
                self.x_l,
                self.base_function_values_1d[k],
                label=rf"$W_{k}$",
            )
        plt.legend()
        plt.show()

    def calc_idx_on_level_l(self, k: int, index: int) -> int:
        """Takes a index on level k and calculates the corresponding index on the finest level l"""
        return int(self.x_li(k, index) / self.h[self.l])

    def alpha_ki(self, base_func_idx, f: Callable, k):
        stencil = np.array([-0.5, 1.0, -0.5])
        x = self.x_li(k, base_func_idx)
        h = self.h[k]
        f_vals = np.array([f(x - h), f(x), f(x + h)])
        return float(stencil @ f_vals)

    def plot_alpha_scaled_base_funcs(self, base_func_idx, index, k):
        # plotting
        start = self.calc_idx_on_level_l(k, base_func_idx - 1)
        stop = self.calc_idx_on_level_l(k, base_func_idx + 1)
        plt.arrow(self.x_li(k, base_func_idx), 0, 0, self.alpha[k, index])
        plt.plot(
            self.x_l[start : stop + 1],
            self.alpha[k, index] * self.base_function_values[k][start : stop + 1],
            label=rf"$\alpha_{{{k},{base_func_idx}}} \cdot W_{k}$",
        )

    def function_approximation(self, f: Callable):
        # 1. get real values of f at the grid points of the finest level
        f_vals = [f(x) for x in self.x_l]
        plt.plot(self.x_l, f_vals, label="f(x)", color="black", linewidth=2)

        self.alpha = np.zeros((self.l + 1, 2**self.l + 1))

        # 2. Calculate hierarchical surpluses alpha_ki
        for k in self.levels:
            I_k = self.indices_of_funcs_on_k(k)
            for base_func_idx in I_k:
                index = self.calc_idx_on_level_l(k, base_func_idx)
                self.alpha[k, index] = self.alpha_ki(base_func_idx, f, k)

                self.plot_alpha_scaled_base_funcs(base_func_idx, index, k)

        plt.legend()
        plt.grid()
        plt.show()

        print("Hierarchical surpluses alpha_ki:\n", self.alpha)

    def alpha_heatmap(self):
        plt.imshow(self.alpha, aspect="auto")
        plt.colorbar(label=r"$\alpha_{k,i}$")
        plt.xlabel("Index i on finest level l")
        plt.ylabel("Level k")
        plt.title("Hierarchical surpluses alpha_ki")
        plt.show()


sg = SparseGrid2d(2)
sg.calculate_base_functions()
# sg.visualize_base_functions()
# sg.function_approximation(lambda x: np.sin(2 * np.pi * x))

# sg.alpha_heatmap()
#
sg.visualize_base_functions_values_2d()
