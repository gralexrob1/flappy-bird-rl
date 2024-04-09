import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_state_values(q_values, title="Flappy Bird State-Value", save_file=None):

    def get_Z(x, y):
        if (x, y) in q_values:
            return max(q_values[(x, y)])
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 14)
        y_range = np.arange(-11, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("State Value")
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(211, projection="3d")
    ax.set_title(title)
    get_figure(ax)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def plot_policy(q_values, title="Flappy Bird Policy", save_file=None):
    """
    Plot the policy for the Flappy Bird environment.

    Args:
        q_values (dict): Dictionary of state-action values.
        title (str): Title of the plot.

    Returns:
        None
    """

    def get_Z(x, y):
        if (x, y) in q_values:
            return np.argmax(q_values[(x, y)])
        else:
            # Return value 2 for unexplored states
            return 2

    def get_figure(ax):
        x_range = np.arange(14, 0, -1)
        y_range = np.arange(-11, 12, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y) for x in x_range] for y in y_range])
        cmap = colors.ListedColormap(["#BEE9E8", "#62B6CB", "#1B4965"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        surf = ax.imshow(Z, cmap=cmap, norm=norm, extent=[0.5, 13.5, -10.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(color="w", linestyle="-", linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, cmap=cmap, norm=norm, ticks=[0, 1, 2], cax=cax)
        cbar.ax.set_yticklabels(["0 (Idle)", "1 (Flap)", "2 (Unexplored)"])

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    get_figure(ax)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
