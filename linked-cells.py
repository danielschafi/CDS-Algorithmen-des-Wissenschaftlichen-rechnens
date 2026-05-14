import numpy as np

rng = np.random.default_rng(42)

# Parameters
EPSILON = 1
SIGMA = 1

# nr of  inner cells per dimension
N_INNER_CELLS = 4
# nr of cells in total per dimension. n inner cells plus boundary cell per side
N_TOTAL = N_INNER_CELLS + 2
# 2.5sigma <= r_c <= 5sigma, size of one cell
R_C = 4 * SIGMA

SIM_AREA = N_INNER_CELLS * R_C


def build_grid(n):
    return [[[] for _ in range(n + 2)] for _ in range(n + 2)]


def populate_grid(grid, n_particles):
    particles = rng.random((n_particles, 2)) * SIM_AREA

    # sort each of the particles into one of the inner cells
    for p in particles:
        bin_x = int(p[0] // R_C) + 1
        bin_y = int(p[1] // R_C) + 1
        print(bin_x)
        grid[bin_x][bin_y] = p

    return grid


def shift_particles(particles, ncol_shift, nrow_shift):
    x_shift = R_C * ncol_shift
    y_shift = R_C * nrow_shift

    for p in particles:
        p[0] = p[0] + x_shift
        p[1] = p[1] + y_shift

    return particles


def populate_boundary(grid):
    for row in range(N_TOTAL):
        for col in range(N_TOTAL):
            # Corners
            is_bl_corner = (col == 0) and (row == 0)
            if is_bl_corner:
                src_cell = grid[col + N_INNER_CELLS][row + N_INNER_CELLS]
                grid[col][row] = shift_particles(
                    src_cell, -N_INNER_CELLS, -N_INNER_CELLS
                )
                continue

            is_br_corner = (col == N_TOTAL) and (row == 0)
            if is_br_corner:
                src_cell = grid[col - N_INNER_CELLS][row + N_INNER_CELLS]
                grid[col][row] = shift_particles(
                    src_cell, +N_INNER_CELLS, -N_INNER_CELLS
                )
                continue

            is_tl_corner = (col == 0) and (row == N_TOTAL)
            if is_tl_corner:
                src_cell = grid[col + N_INNER_CELLS][row - N_INNER_CELLS]
                grid[col][row] = shift_particles(
                    src_cell, -N_INNER_CELLS, +N_INNER_CELLS
                )
                continue

            is_tr_corner = (col == N_TOTAL) and (row == N_TOTAL)
            if is_tr_corner:
                src_cell = grid[col - N_INNER_CELLS][row - N_INNER_CELLS]
                grid[col][row] = shift_particles(
                    src_cell, +N_INNER_CELLS, +N_INNER_CELLS
                )
                continue

            # sides
            is_left = col == 0

            is_right = col == N_TOTAL

            is_bottom = row == 0

            is_top = row == N_TOTAL


# sort particels into cells

# list etc including boundary cells, just need to sort them into the right cells then


import matplotlib.pyplot as plt


def plot_grid_with_particles(grid):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Cell grid lines
    for k in range(N_INNER_CELLS + 1):
        ax.axhline(k * R_C, color="gray", linewidth=0.6)
        ax.axvline(k * R_C, color="gray", linewidth=0.6)

    # plt particles
    for row in range(N_TOTAL):
        for col in range(N_TOTAL):
            points = grid[row][col]
            if len(points) > 0:
                ax.scatter(
                    points[0],
                    points[1],
                    color="black",
                )

    plt.show()


def main():
    grid = build_grid(N_INNER_CELLS)

    grid = populate_grid(grid, 40)

    plot_grid_with_particles(grid)


if __name__ == "__main__":
    main()
