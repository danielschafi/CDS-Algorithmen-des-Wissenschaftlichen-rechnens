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

        # nd operator matrix
        self.op_matrix = self.get_operator_matrix()

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

    def get_operator_matrix(self, dim: int = 2):
        op_1d = np.array([-0.5, 1, -0.5])
        op_nd = op_1d
        for d in range(dim - 1):
            op_nd = np.tensordot(op_1d, op_nd, axes=0)
        return op_nd

    def alpha_ki(self, i1: int, i2: int, f: Callable, k: int) -> float:
        """
        Hierarchical surplus at grid point (x_i1, x_i2) on level k,
        computed via the 2D tensor-product stencil [-0.5, 1, -0.5] ⊗ [-0.5, 1, -0.5].
        """
        x_1 = self.x_li(k, i1)
        x_2 = self.x_li(k, i2)
        h = self.h[k]
        f_vals = np.array(
            [
                [f(x_1 - h, x_2 - h), f(x_1, x_2 - h), f(x_1 + h, x_2 - h)],
                [f(x_1 - h, x_2), f(x_1, x_2), f(x_1 + h, x_2)],
                [f(x_1 - h, x_2 + h), f(x_1, x_2 + h), f(x_1 + h, x_2 + h)],
            ]
        )
        return float(np.sum(self.op_matrix * f_vals))

    def evaluate(self, x: float, y: float) -> float:
        """
        Evaluate the sparse grid approximation u(x, y) at a point.
        u(x,y) = sum_{k,i1,i2} alpha[k,i1,i2] * phi_li(x,k,i1) * phi_li(y,k,i2)
        """
        result = 0.0
        for k in self.levels:
            for i1 in self.indices_of_funcs_on_k(k):
                phi_x = self.phi_li(x, k, i1)
                if phi_x == 0.0:
                    continue
                idx1 = self.calc_idx_on_level_l(k, i1)
                for i2 in self.indices_of_funcs_on_k(k):
                    idx2 = self.calc_idx_on_level_l(k, i2)
                    result += self.alpha[k, idx1, idx2] * phi_x * self.phi_li(y, k, i2)
        return result

    def function_approximation_2d(self, f: Callable):
        """
        Compute hierarchical surpluses and plot the approximation vs. the true function.
        """
        self.alpha = np.zeros((self.l + 1, 2**self.l + 1, 2**self.l + 1))

        # Compute hierarchical surpluses alpha_{k,i1,i2}
        for k in self.levels:
            for i1 in self.indices_of_funcs_on_k(k):
                for i2 in self.indices_of_funcs_on_k(k):
                    idx1 = self.calc_idx_on_level_l(k, i1)
                    idx2 = self.calc_idx_on_level_l(k, i2)
                    self.alpha[k, idx1, idx2] = self.alpha_ki(i1, i2, f, k)

        # Evaluate approximation on the finest grid
        X, Y = np.meshgrid(self.x_l, self.x_l)
        f_real = np.vectorize(f)(X, Y)
        u_approx = np.zeros_like(X)
        for i, xi in enumerate(self.x_l):
            for j, yj in enumerate(self.x_l):
                u_approx[i, j] = self.evaluate(xi, yj)

        # Plot real vs. approximation side by side
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))

        axes[0].plot_surface(X, Y, f_real, alpha=0.8)
        axes[0].set_title("f(x, y)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        axes[1].plot_surface(X, Y, u_approx, alpha=0.8)
        axes[1].set_title(f"Sparse Grid Approximation (l={self.l})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        plt.suptitle(f"Sparse Grid 2D Approximation (l={self.l})")
        plt.tight_layout()
        plt.show()

        # Also show the error
        fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 6))
        ax2.plot_surface(X, Y, np.abs(f_real - u_approx), alpha=0.8)
        ax2.set_title(f"|f - u| (l={self.l})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.tight_layout()
        plt.show()

    def alpha_heatmap(self):
        # Sum over levels for a 2D view
        alpha_sum = np.sum(self.alpha, axis=0)
        plt.imshow(alpha_sum, aspect="auto", origin="lower", extent=[0, 1, 0, 1])
        plt.colorbar(label=r"$\sum_k \alpha_{k,i_1,i_2}$")
        plt.xlabel("Index $i_2$ (dim 2)")
        plt.ylabel("Index $i_1$ (dim 1)")
        plt.title("Hierarchical surpluses (summed over levels)")
        plt.show()


sg = SparseGrid2d(3)
sg.calculate_base_functions()
sg.function_approximation_2d(lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y))
sg.visualize_base_functions_values_2d()
