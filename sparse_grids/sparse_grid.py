import matplotlib.pyplot as plt
import numpy as np


class SparseGrid:
    def __init__(self, l: int):
        self.l: int = l  # Total layer count

        self.levels = list(range(1, self.l + 1))
        print(f"levels: {self.levels}")
        # Grid spacing per level
        self.h = np.array(
            [2 ** (-k) for k in range(self.l + 1)]
        )  # get grid spacing for levels 0 to l, we will only use levels 1 to l, but this way we have the correct index for h[k]
        print(f"Grid spacing per level: {self.h}")
        # Base function values.
        self.base_function_values = np.zeros((self.l + 1, 2**self.l + 1))

        print(f"Base function values shape: {self.base_function_values.shape}")
        print(self.base_function_values)

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

    def calculate_base_functions(self):
        """
        Builds the base functions for all levels and saves the values at the resolution of the finest level.
        """

        x_l = np.linspace(0, 1, 2**self.l + 1)

        for k in self.levels:
            I_k = self.indices_of_funcs_on_k(k)
            print(I_k)

            for base_func_idx in I_k:
                for i in range(len(x_l)):
                    self.base_function_values[k, i] += self.phi_li(
                        float(x_l[i]), k, base_func_idx
                    )

    def visualize_base_functions(self):
        """
        Plots the base functions on different levels.
        """
        print(self.base_function_values)
        plt.title("Base functions on different levels")
        for k in self.levels:
            plt.plot(
                np.linspace(0, 1, 2**self.l + 1),
                self.base_function_values[k],
                label=rf"$W_{k}$",
            )
        plt.legend()
        plt.show()


sg = SparseGrid(4)
sg.calculate_base_functions()
sg.visualize_base_functions()
