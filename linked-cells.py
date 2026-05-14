import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
        grid[bin_x][bin_y].append(p)

    return grid


def shift_particles(particles, ncol_shift, nrow_shift):
    x_shift = R_C * ncol_shift
    y_shift = R_C * nrow_shift
    return [np.array([p[0] + x_shift, p[1] + y_shift]) for p in particles]


def populate_boundary(grid):
    for col in range(N_TOTAL):
        for row in range(N_TOTAL):
            # src_cell refers to the cell that is going to be copied from

            # Corners
            is_bl_corner = (col == 0) and (row == 0)
            if is_bl_corner:
                src_cell = grid[col + N_INNER_CELLS][row + N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(
                    src_cell, -N_INNER_CELLS, -N_INNER_CELLS
                )
                continue

            is_br_corner = (col == N_TOTAL - 1) and (row == 0)
            if is_br_corner:
                src_cell = grid[col - N_INNER_CELLS][row + N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(
                    src_cell, +N_INNER_CELLS, -N_INNER_CELLS
                )
                continue

            is_tl_corner = (col == 0) and (row == N_TOTAL - 1)
            if is_tl_corner:
                src_cell = grid[col + N_INNER_CELLS][row - N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(
                    src_cell, -N_INNER_CELLS, +N_INNER_CELLS
                )
                continue

            is_tr_corner = (col == N_TOTAL - 1) and (row == N_TOTAL - 1)
            if is_tr_corner:
                src_cell = grid[col - N_INNER_CELLS][row - N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(
                    src_cell, +N_INNER_CELLS, +N_INNER_CELLS
                )
                continue

            # sides
            is_left = col == 0
            if is_left:
                src_cell = grid[col + N_INNER_CELLS][
                    row
                ]  # take from rightmost inner cell on same row
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(
                    src_cell, -N_INNER_CELLS, 0
                )  # shift to left boundary
                continue

            is_right = col == N_TOTAL - 1
            if is_right:
                src_cell = grid[col - N_INNER_CELLS][row]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(src_cell, +N_INNER_CELLS, 0)
                continue

            is_bottom = row == 0
            if is_bottom:
                src_cell = grid[col][row + N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(src_cell, 0, -N_INNER_CELLS)
                continue

            is_top = row == N_TOTAL - 1
            if is_top:
                src_cell = grid[col][row - N_INNER_CELLS]
                # print(f"nr of particles: {len(src_cell)}, - {src_cell}")
                grid[col][row] = shift_particles(src_cell, 0, +N_INNER_CELLS)
                continue
    return grid


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def find_neighbours(grid):
    """
    Get pairs of points that are within r_c of each other
    """

    pairs = []

    for col in range(N_TOTAL):
        for row in range(N_TOTAL):
            # only consider cells with higher indices

            # check only higher indices. bthanks newtons 3d law.
            candidates = []
            if col + 1 <= N_TOTAL - 1:
                candidates.extend(grid[col + 1][row])
                if row + 1 <= N_TOTAL - 1:
                    candidates.extend(grid[col + 1][row + 1])
            if row + 1 <= N_TOTAL - 1:
                candidates.extend(grid[col][row + 1])

            candidates.extend(grid[col][row])
            for particle in grid[col][row]:
                for candidate_particle in candidates:
                    if dist(particle, candidate_particle) <= R_C:
                        pairs.append((particle, candidate_particle))

    return pairs


def plot_grid_with_particles(grid, pairs=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Gray background for boundary/ghost region; white for inner simulation area
    ax.set_facecolor("lightgray")
    ax.add_patch(plt.Rectangle((0, 0), SIM_AREA, SIM_AREA, color="white", zorder=0))

    # Grid lines spanning the full extent (inner + one ghost cell each side)
    ticks = np.arange(-R_C, SIM_AREA + R_C + 1e-9, R_C)
    for t in ticks:
        ax.axhline(t, color="gray", linewidth=0.6, zorder=1)
        ax.axvline(t, color="gray", linewidth=0.6, zorder=1)

    # Plot particles
    for row in range(N_TOTAL):
        for col in range(N_TOTAL):
            points = grid[col][row]
            if points:
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                ax.scatter(x, y, color="black", s=20, zorder=2)

    if pairs is not None:
        # plot the neighbors
        for pair in pairs:
            ax.add_line(
                Line2D(
                    [pair[0][0], pair[1][0]],
                    [pair[0][1], pair[1][1]],
                    color="red",
                    linewidth=1,
                    zorder=1.5,
                )
            )

    ax.set_xlim(-R_C, SIM_AREA + R_C)
    ax.set_ylim(-R_C, SIM_AREA + R_C)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    plt.tight_layout()
    # plt.savefig("linked-with-neighbours.png")
    plt.show()


def main():
    grid = build_grid(N_INNER_CELLS)

    grid = populate_grid(grid, 40)
    grid = populate_boundary(grid)

    pairs = find_neighbours(grid)
    plot_grid_with_particles(grid, pairs)


if __name__ == "__main__":
    main()
