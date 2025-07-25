import numpy as np
import matplotlib.pyplot as plt

def generate_circle_points(n_points=100):
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.vstack((x, y)).T  # shape (n_points, 2)

def generate_spiral_points(n_points=100, a=0.1):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = a * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack((x, y)).T
