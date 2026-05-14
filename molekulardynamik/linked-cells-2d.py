import matplotlib.pyplot as plt
import numpy as np


def build_grid(n, rc=1.0):
    """
    Build an (n+2) x (n+2) cell grid (inner cells + 1 boundary layer).
    Each cell stores a list of particle indices.
    Returns the grid and the periodic boundary references.

    Cell (i,j): i=row (0=bottom halo), j=col (0=left halo)
    Inner cells: rows 1..n, cols 1..n
    """
    grid = [[[] for _ in range(n + 2)] for _ in range(n + 2)]
    return grid


def insert_particles(grid, positions, n, rc=1.0):
    """Place each particle into the correct inner cell."""
    for pid, (x, y) in enumerate(positions):
        ci = int(y / rc) + 1  # row index (1-based, inner)
        cj = int(x / rc) + 1  # col index (1-based, inner)
        ci = min(max(ci, 1), n)
        cj = min(max(cj, 1), n)
        grid[ci][cj].append(pid)
    return grid


def populate_halo(grid, n):
    """
    Copy particle lists from inner edge cells into the halo cells on the
    opposite side, implementing periodic boundary conditions (slide 33).

    Layout (n=4 example, total size 6x6):
      row 0   = bottom halo  → mirrors inner row n   (top edge)
      row n+1 = top halo     → mirrors inner row 1   (bottom edge)
      col 0   = left halo    → mirrors inner col n   (right edge)
      col n+1 = right halo   → mirrors inner col 1   (left edge)
    """
    for i in range(n + 2):
        grid[0][i] = grid[n][i][:]  # bottom halo ← top inner row
        grid[n + 1][i] = grid[1][i][:]  # top halo    ← bottom inner row
    for i in range(n + 2):
        grid[i][0] = grid[i][n][:]  # left halo   ← right inner col
        grid[i][n + 1] = grid[i][1][:]  # right halo  ← left inner col


def find_neighbours(grid, positions, n, rc=1.0):
    """
    For every particle find all neighbours within rc, including those
    reached via the halo (periodic boundaries).

    Only checks the 5 offsets with higher flat index so each pair is
    counted once (Newton's 3rd law, slide 34).

    When the neighbour sits in a halo cell the distance is computed with
    a periodic correction so the geometry is correct across the boundary.
    """
    domain = n * rc
    pairs = []

    for ci in range(1, n + 1):
        for cj in range(1, n + 1):
            # 5 offsets that cover all unique pairs (same cell + 4 forward)
            for di, dj in [(0, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]:
                ni, nj = ci + di, cj + dj

                # Allow halo indices (0 and n+1) but nothing beyond
                if not (0 <= ni <= n + 1 and 0 <= nj <= n + 1):
                    continue

                for p in grid[ci][cj]:
                    for q in grid[ni][nj]:
                        if p == q:
                            continue
                        if ni == ci and nj == cj and q <= p:
                            continue

                        dx = positions[q][0] - positions[p][0]
                        dy = positions[q][1] - positions[p][1]

                        # Minimum-image correction for periodic distance
                        if dx > domain / 2:
                            dx -= domain
                        if dx < -domain / 2:
                            dx += domain
                        if dy > domain / 2:
                            dy -= domain
                        if dy < -domain / 2:
                            dy += domain

                        if dx * dx + dy * dy <= rc * rc:
                            pairs.append((p, q))
    return pairs


def run(n=4, N=30, rc=1.0, seed=42):
    """
    n  : number of inner cells per dimension
    N  : number of particles
    rc : cutoff radius (= cell side length)
    """
    rng = np.random.default_rng(seed)
    domain = n * rc  # physical domain size

    # Random positions inside the domain
    positions = rng.uniform(0, domain, size=(N, 2))

    # Build grid, insert particles, populate halo
    grid = build_grid(n, rc)
    grid = insert_particles(grid, positions, n, rc)
    populate_halo(grid, n)  # copies edge cells into halo layer

    # Find neighbour pairs (now includes cross-boundary pairs)
    pairs = find_neighbours(grid, positions, n, rc)

    # Separate regular vs periodic (cross-boundary) pairs for plotting
    regular_pairs = []
    periodic_pairs = []
    for p, q in pairs:
        dx = abs(positions[q][0] - positions[p][0])
        dy = abs(positions[q][1] - positions[p][1])
        if dx > domain / 2 or dy > domain / 2:
            periodic_pairs.append((p, q))
        else:
            regular_pairs.append((p, q))

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))

    # Cell grid lines
    for k in range(n + 1):
        ax.axhline(k * rc, color="#B4B2A9", linewidth=0.6)
        ax.axvline(k * rc, color="#B4B2A9", linewidth=0.6)

    # Regular neighbour connections
    for p, q in regular_pairs:
        x0, y0 = positions[p]
        x1, y1 = positions[q]
        ax.plot([x0, x1], [y0, y1], color="#85B7EB", linewidth=0.8, alpha=0.7, zorder=1)

    # Periodic (cross-boundary) connections — dashed orange
    for p, q in periodic_pairs:
        x0, y0 = positions[p]
        x1, y1 = positions[q]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#EF9F27",
            linewidth=0.9,
            alpha=0.8,
            zorder=1,
            linestyle="--",
        )

    # Particles
    ax.scatter(positions[:, 0], positions[:, 1], s=40, color="#D85A30", zorder=3)

    ax.set_xlim(0, domain)
    ax.set_ylim(0, domain)
    ax.set_aspect("equal")
    ax.set_title(
        f"Linked-Cells 2D  |  n={n}, N={N}, rc={rc}\n"
        f"{len(regular_pairs)} regular + {len(periodic_pairs)} periodic pairs",
        fontsize=11,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Legend
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color="#85B7EB", lw=1.5, label="regular neighbour"),
            Line2D(
                [0],
                [0],
                color="#EF9F27",
                lw=1.5,
                ls="--",
                label="periodic (halo) neighbour",
            ),
        ],
        fontsize=9,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("linked_cells.png", dpi=150)
    plt.show()
    print(
        f"Grid: {n}x{n} inner cells, {N} particles, "
        f"{len(regular_pairs)} regular + {len(periodic_pairs)} periodic pairs"
    )


if __name__ == "__main__":
    run(n=4, N=30, rc=1.0)
