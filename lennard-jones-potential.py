import matplotlib.pyplot as plt
import numpy as np


def U_ij(r_ij, epsilon=1, sigma=1) -> float:
    x = (sigma / r_ij) ** 6
    return 4 * epsilon * x * (x - 1)


def F_ij(r_ij, epsilon=1, sigma=1) -> float:
    x = (sigma / r_ij) ** 6
    return (24 * epsilon / r_ij) * x * (2 * x - 1)


particle_distances = np.linspace(0, 3, 1000)

lennard_jones_potentials = U_ij(r_ij=particle_distances)
lennard_jones_forces = F_ij(r_ij=particle_distances)


plt.plot(particle_distances, lennard_jones_potentials, label="U_lj")

plt.plot(particle_distances, lennard_jones_forces, label="F_lj")
plt.ylim((-3, 3))
plt.grid()
plt.legend()
plt.savefig("lennard-jones-potential.png")

plt.show()
