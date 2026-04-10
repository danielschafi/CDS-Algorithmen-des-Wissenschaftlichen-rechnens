import numpy as np


class Particle:
    x: float
    y: float


def lj(r_ij, epsilon, sigma):
    x = (sigma / r_ij) ** 6

    r_c = 5 * sigma

    if r_ij <= r_c:
        return (24 * epsilon / r_ij) * x * (2 * x - 1)
    else:
        print("dfgd")
        return 0


import matplotlib.pyplot as plt


def main():
    p_i = np.array([0.2, 0.3])
    p_j = np.array([0.5, 0.6])

    epsilon = 1
    sigma = 1

    # r_ij = np.linalg.norm(p_i - p_j)

    x = np.linspace(1, 3, 1000)
    ljp = []
    for r_ij in x:
        ljp.append(lj(r_ij, epsilon, sigma))

    plt.plot(x, ljp)
    plt.show()
    # potenzial = lj(r_ij, epsilon, sigma)
    # print(potenzial)


if __name__ == "__main__":
    main()
