import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    x = 1.0 / np.arange(2.0, 6.0, 0.1)

    fig, ax = plt.subplots()

    ax.plot(x - 1.0 / 3.1, 1 + 30.0 * x ** 2, "k-")

    plt.show()
