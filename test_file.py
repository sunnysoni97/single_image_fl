import numpy as np

if __name__ == "__main__":

    grid_size = 21
    x_coords = np.linspace(0, 1.0, grid_size)
    print(x_coords)
    x_coords = (x_coords[:-1] + np.roll(x_coords, -1)[:-1])/2
    print(x_coords)
