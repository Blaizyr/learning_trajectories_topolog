from matplotlib import pyplot as plt

from ml.generate_.geometrical.generate_geometrical import generate_circle_points, generate_spiral_points


def represent():
    circle = generate_circle_points()
    spiral = generate_spiral_points()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Circle")
    plt.scatter(circle[:, 0], circle[:, 1])
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title("Spiral")
    plt.scatter(spiral[:, 0], spiral[:, 1])
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    represent()
