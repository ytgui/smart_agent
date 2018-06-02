import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, transform


def plot3d_1():
    X, Y = np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')

    plt.show()


def plot3d_2():
    image = io.imread('1.bmp')
    image = transform.rescale(image, scale=0.5)
    h, w = image.shape

    x, y = np.arange(0, w, 1), np.arange(0, h, 1)
    x, y = np.meshgrid(x, y)

    image = image * 0.1
    image[40:45, 40:45] = 0.6

    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    ax.plot_surface(x, y, image, rstride=1, cstride=1, cmap='brg', alpha=1.0)
    # ax.bar3d(50, 50, 2, 10, 10, 8, color='green', alpha=0.8, zorder=0.5, shade=True)
    ax.set_zlim(top=2.5)

    plt.show()


def plot3d_3():
    image = io.imread('1.bmp')
    image = transform.rescale(image, scale=0.2, mode='reflect') > 0.75
    h, w = image.shape

    x, y = np.arange(0, w, 1), np.arange(0, h, 1)
    x, y = np.meshgrid(x, y)
    x, y, image = x.ravel(), y.ravel(), image.ravel()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.bar3d(x, y, np.zeros_like(image), 1, 1, image)
    ax.bar3d(10, 15, 0, 2, 2, 8, color='green')
    ax.set_zlim(top=10)

    plt.show()


def main():
    plot3d_2()


if __name__ == '__main__':
    main()
