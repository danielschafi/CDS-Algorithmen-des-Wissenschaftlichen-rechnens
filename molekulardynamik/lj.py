import matplotlib.pyplot as plt
import numpy as np


def lj_force(r, epsilon=1.0, sigma=1.0):
    """
    Compute the Lennard-Jones force magnitude F_LJ(r).

    F_LJ(r) = (24 * epsilon / r) * (2*(sigma/r)^12 - (sigma/r)^6)

    Positive F means repulsive (atoms push apart).
    Negative F means attractive (atoms pull together).

    Parameters
    ----------
    r       : float or ndarray  – interparticle distance
    epsilon : float             – energy parameter  [eV or reduced units]
    sigma   : float             – size parameter    [Å  or reduced units]

    Returns
    -------
    F_LJ : same shape as r
    """
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return (24.0 * epsilon / r) * (2.0 * sr12 - sr6)


def lj_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute the Lennard-Jones potential U_LJ(r).

    U_LJ(r) = 4 * epsilon * ((sigma/r)^12 - (sigma/r)^6)

    Parameters
    ----------
    r       : float or ndarray
    epsilon : float
    sigma   : float

    Returns
    -------
    U_LJ : same shape as r
    """
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return 4.0 * epsilon * (sr12 - sr6)


def plot_lj(epsilon=1.0, sigma=1.0, r_min=0.9, r_max=3.0, n=500):
    """Plot U_LJ and F_LJ on the same axes."""
    r = np.linspace(r_min, r_max, n)

    U = lj_potential(r, epsilon, sigma)
    F = lj_force(r, epsilon, sigma)

    # Clamp for readability (very large values near r→0 distort the plot)
    clip = 5.0
    U_plot = np.clip(U, -clip, clip)
    F_plot = np.clip(F, -clip, clip)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(r, U_plot, color="#D85A30", linewidth=2, label=r"$U_{LJ}(r)$ – potential")
    ax.plot(r, F_plot, color="#185FA5", linewidth=2, label=r"$F_{LJ}(r)$ – force")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Mark the equilibrium distance r_eq = 2^(1/6) * sigma
    r_eq = 2.0 ** (1.0 / 6.0) * sigma
    ax.axvline(r_eq, color="gray", linewidth=0.8, linestyle=":")
    ax.annotate(
        rf"$r_{{eq}} = 2^{{1/6}}\sigma \approx {r_eq:.3f}$",
        xy=(r_eq, 0),
        xytext=(r_eq + 0.15, 0.6),
        fontsize=9,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    ax.set_xlim(r_min, r_max)
    ax.set_ylim(-clip, clip)
    ax.set_xlabel(r"$r_{ij}$ (reduced units)", fontsize=12)
    ax.set_ylabel("Energy / Force (reduced units)", fontsize=12)
    ax.set_title(
        rf"Lennard-Jones potential & force  ($\varepsilon = {epsilon}$, $\sigma = {sigma}$)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lennard_jones.png", dpi=150)
    plt.show()
    print("Plot saved to lennard_jones.png")


if __name__ == "__main__":
    # ── Aufgabe 1 test: ε = σ = 1 ──────────────────────────────────────────
    eps, sig = 1.0, 1.0

    print("=== Lennard-Jones verification (ε = σ = 1) ===")
    for r_test in [0.95, 1.0, 1.122, 1.5, 2.0, 2.5]:
        U = lj_potential(r_test, eps, sig)
        F = lj_force(r_test, eps, sig)
        print(f"  r = {r_test:.3f}  →  U = {U:+8.4f}  F = {F:+8.4f}")

    print("\n  Minimum of U at r_eq = 2^(1/6) ≈", round(2 ** (1 / 6), 6))
    print(
        "  U(r_eq) =",
        round(lj_potential(2 ** (1 / 6), eps, sig), 6),
        "  (should be -1)",
    )
    print("  F(r_eq) =", round(lj_force(2 ** (1 / 6), eps, sig), 8), "  (should be ~0)")

    plot_lj(epsilon=eps, sigma=sig)
