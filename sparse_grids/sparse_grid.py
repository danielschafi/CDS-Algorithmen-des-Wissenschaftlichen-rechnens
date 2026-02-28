import matplotlib.pyplot as plt
import numpy as np


class SparseGrid:
    def __init__(self, l: int):
        self.l: int = l  # Total layer count

        # Grid spacing per level
        self.h = np.array([2 ** (-k) for k in range(self.l)])

        # Base function values.
        self.base_function_values = np.zeros((self.l, 2**self.l + 1))

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

    def phi_li(self, x: float | np.float64, k: int, i: int, h_l: float) -> float:
        """
        Base function for gridpoint x_li(l, i) on level l
        with phi_li(x_li) = 1 on [x_li - h_l, x_li + h_l]. 1 in the center and 0 at the borders

        Parameters
            x: (float) x coordinate
            k: (int) level
            i: (int) index of the grid point of the base function (on level k)
            h_l: (float) grid spacing on level l
        Returns
            float value of the base function at x
        """
        return self.phi((x - self.x_li(k, i)) / h_l)

    def indices_of_funcs_on_k(self, k: int) -> list[int]:
        """
        Indexset. I_k = {i \in {1,..., 2**k-1} | i is odd}

        Calculates the indexes of the base functions on level k.
        On each level the base functions are centered on the uneven indexes
        and have support on [x_li(k, i) - h[k], x_li(k, i) + h[k]]
        """
        return [i for i in range(1, 2**k) if i % 2 == 1]

    def calculate_base_functions(self):
        """
        Builds the base functions for all levels and saves the values at the resolution of the finest level.
        """

        # Each row in marix corresponds to a level W_k.

        # X coords on the finest level, we have to calculate the values of the base function at these points.
        x_l = np.linspace(0, 1, 2**self.l + 1)

        for k in range(1, self.l):
            I_k = self.indices_of_funcs_on_k(k)

            for base_func_idx in I_k:
                for i in range(2**self.l + 1):
                    self.base_function_values[k, i] = self.phi_li(
                        float(x_l[i]), k, base_func_idx, self.h[k]
                    )

    def visualize_base_functions(self):
        """
        Plots the base functions on different levels.
        """
        plt.title("Base functions on different levels")
        for k in range(1, self.l):
            plt.plot(
                np.linspace(0, 1, 2**self.l + 1),
                self.base_function_values[k],
                label=rf"$W_{k}$",
            )
        plt.legend()
        plt.show()


sg = SparseGrid(2)
sg.calculate_base_functions()
sg.visualize_base_functions()
