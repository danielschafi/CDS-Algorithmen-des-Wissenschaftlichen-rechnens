import itertools
from typing import Callable, Collection, List

import matplotlib.pyplot as plt
import numpy as np


class SparseGridNd:
    """
    Class to approximate functions using sparse grids for n-dimensional functions
    """

    def __init__(self, depth: int, dimension: int) -> None:
        # number of hierarchical layers to use for approximations
        self.l: int = depth

        # dimension of the function to be approximated
        self.dim: int = dimension

        # List of levels relevant in approximation +1 bc. range is exclusive stop
        self.levels: List[int] = list(range(1, self.l + 1))

        # Grid spacing on each level
        self.h: np.ndarray = np.array([2 ** (-k) for k in range(self.l + 1)])

        # Storage for the base function values in n-dim
        self.storage_size = [self.l + 1] + [2**self.l + 1] * self.dim
        print(f"storage_size {self.storage_size}")
        self.base_function_values = np.zeros(self.storage_size)

        # X points on finest grid
        self.x_l = np.linspace(0, 1, 2**self.l + 1)

        self.op_matrix: np.ndarray = self.get_operator_matrix()

    ################################################
    ################     Utility    ################
    ################################################

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

    ################################################
    ###########     Base Functions    ##############
    ################################################

    def calculate_base_functions(self) -> None:
        """
        Builds the base functions for all levels and saves the values at the resolution of the finest level.

        Assumes equal grid spacing and equal depth across all dimensions
        """

        bf_1d: Collection[float] = np.zeros((self.l + 1, 2**self.l + 1))

        for k in self.levels:
            I_k = self.indices_of_funcs_on_k(k)

            for base_func_idx in I_k:
                for i in range(len(self.x_l)):
                    bf_1d[k, i] += self.phi_li(float(self.x_l[i]), k, base_func_idx)

            # To get the base function values in the required dimension
            # Form the Tensorproduct self.dim times
            bf_nd = bf_1d[k]
            for _ in range(self.dim - 1):
                bf_nd = np.tensordot(bf_nd, bf_1d[k], axes=0)
            self.base_function_values[k] = bf_nd

    def visualize_base_function_values(self) -> None:
        """Visualize the base functions on lowest level"""
        if self.dim == 1:
            plt.title(f"Base functions of levels 1-{self.l}")
            for k in self.levels:
                plt.plot(
                    self.x_l,
                    self.base_function_values[k],
                    label=rf"$W_{k}$",
                )
            plt.legend()
            plt.show()

        elif self.dim == 2:
            X, Y = np.meshgrid(self.x_l, self.x_l)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            surface = ax.plot_surface(
                X, Y, self.base_function_values[self.l], alpha=0.8
            )
            fig.colorbar(surface)
            plt.title("Base function values of level l")
            plt.show()

        elif self.dim > 2:
            raise NotImplementedError(
                "Cant visualize more than 3d sadly, let me know if you figure it out."
            )

    ################################################
    #############     Approximation    #############
    ################################################

    def calc_idx_on_level_l(self, k: int, index: int) -> int:
        """Takes a index on level k and calculates the corresponding index on the finest level l"""
        return int(self.x_li(k, index) / self.h[self.l])

    def get_operator_matrix(self) -> np.ndarray:
        """
        Builds the operator matrix for the calculation of the alphas
        """
        op_1d = np.array([-0.5, 1, -0.5])
        op_nd = op_1d
        for _ in range(self.dim - 1):
            op_nd = np.tensordot(op_1d, op_nd, axes=0)
        return op_nd

    def alpha_ki(self, f: Callable, k: int, grid_indexes: List[int]):
        """
        Hierarchical surplus as gridpoint (x,y) on level k.
        Where x,y coordinates are calculated based on the indexes

        Parameters
            f: the function to approximate
            k: the current level
            grid_indexes: the index where to calculate the hierarchical surpluss. needs to have a value for each dimension
        [x1,x2,...]
        """
        if self.dim != len(grid_indexes):
            raise ValueError(
                "dimension of function f needs to be equal to the length of grid indices, provide a value for each dimension"
            )

        h = self.h[k]
        offsets = [-h, 0.0, h]

        f_vals: List[float] = []

        center = np.array([self.x_li(k, i) for i in grid_indexes])
        # need to get the function values in a matrix
        # 3 values for each dimension x-h, x, x+h
        # we get all ofset combinations
        center_cpy = center.copy()
        print("Center point: ", center)
        for offset_idxs in itertools.product(range(3), repeat=self.dim):
            # yields all offset combinations: 00, 01, 10, 11, 02, 20, etc.
            center = center_cpy.copy()

            for idx, val in enumerate(offset_idxs):
                center[idx] += offsets[val]

            f_val = f(*center)
            f_vals.append(f_val)

            print(f"Offset: {offset_idxs},\t Shifted Point: {center},\t f: {f_val}")

        hierarchical_surplus = np.sum(np.array(f_vals) * self.op_matrix.flatten())

        return float(hierarchical_surplus)

    def function_approximation(self, f: Callable) -> None:
        self.alpha = np.zeros(self.storage_size)

        # calculate hierarchical surplusses and store them
        for k in self.levels:
            bfi = self.indices_of_funcs_on_k(k)
            for multi_idx in itertools.product(bfi, repeat=self.dim):
                idx_level_l = [self.calc_idx_on_level_l(k, idx) for idx in multi_idx]
                self.alpha[k, *idx_level_l] = self.alpha_ki(f, k, list(multi_idx))

    def evaluate(self, coordinates: List[float]) -> float:
        """
        Perform the sparse grid approximation at u(x,y)-
        """
        result = 0
        for k in self.levels:
            # 1. Evaluate ALL base functions on level k on that point
            # For that we need to get all the indices of the base functions
            # that is
            # all possible combinations of the basefunc indices on k
            # meshgrid in 2d
            # ccartesian product n times

            bfi = self.indices_of_funcs_on_k(k)

            # all the positions of the base functions in all dimensions
            for multi_idx in itertools.product(bfi, repeat=self.dim):
                idx_level_l = [self.calc_idx_on_level_l(k, idx) for idx in multi_idx]
                phis = [
                    self.phi_li(x, k, idx) for x, idx in zip(coordinates, multi_idx)
                ]
                result += self.alpha[k, *idx_level_l] * np.prod(phis)

        return result


def main():
    dim = 2
    sg = SparseGridNd(depth=3, dimension=dim)
    sg.calculate_base_functions()

    # sg.visualize_base_function_values()
    # print(sg.base_function_values)
    def d1(x):
        return np.sin(2 * np.pi * x)

    def d2(x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def d3(x, y, z):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)

    funcs = [d1, d2, d3]

    sg.alpha_ki(funcs[dim - 1], 2, [2] * dim)

    sg.function_approximation(funcs[dim - 1])
    sg.evaluate([2] * dim)


if __name__ == "__main__":
    main()
