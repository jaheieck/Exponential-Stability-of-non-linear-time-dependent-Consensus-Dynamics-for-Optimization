import os

import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
import math

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


def _set_default_rc_params():
    plt.rcParams.update({'figure.figsize': (8, 8)})
    plt.ioff()


# 1-dimensional visualization of all particle movements
def visualize_trajectory_1d(trajectory, function, dt, output_path=None, x_range=None):
    _set_default_rc_params()
    if output_path is None:
        output_path = 'trajectory.gif'

    if x_range is None:
        x_range = (-10, 10)
    xs = torch.linspace(*x_range, 1000)
    function_values = function(xs.reshape(-1, 1))
    images = []

    for i, trajectory_point in enumerate(trajectory):
        V, V_alpha, V_best = trajectory_point['V'], trajectory_point['V_alpha'], trajectory_point['V_best']

        plt.clf()
        plt.title(f't = {(dt * i):.2f}, V_alpha = {V_alpha.item():.2f}')
        plt.ylim([function_values.min(), function_values.max()])
        plt.xlim([x_range[0], x_range[1]])

        plt.plot(xs, function_values, color='red', alpha=1, zorder=1)
        plt.scatter(V, function(V).reshape(-1,1), marker='o', color='black', label='particles', zorder=2)
        plt.scatter([V_best], [function(V_best.reshape(-1, 1))], marker='o', color='blue', label='best particle', zorder=3)
        plt.scatter([V_alpha], [function(V_alpha.reshape(-1, 1))], marker='o', color='green', label='consensus', zorder=4)

        plt.legend()

        plot_path = os.path.join(os.path.dirname(output_path), f'{i}_tmp.png')
        plt.savefig(plot_path)
        image = iio.imread(plot_path)
        images.append(image)
        os.remove(plot_path)

    plt.clf()
    if os.path.exists(output_path):
        # Is required for a correct gifs rendering in notebooks
        os.remove(output_path)
    imageio.mimsave(output_path, images)

    plt.subplots_adjust(top=0.9)  # Adjust this value to reduce the top space

    # safe plot
    output_folder = 'images'  # name of image folder
    os.makedirs(output_folder, exist_ok=True)  # create folder if not already existing
    output_path = os.path.join(output_folder, 'rastrigin_plot_3d.png')  # safe plot in folder
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.1)  

    plt.show()
    plt.close()  # close plot




# 2-dimensional visualization of all particle movement with snapshots
def visualize_trajectory_2d(trajectory, function, dt, output_path=None, x_range=None, y_range=None,
                            elev_deg=None, azim_deg=None, screenshot_times=None, screenshot_dir=None, summary_title=None):

    if output_path is None:
        output_path = 'trajectory_2D.gif'
    if screenshot_times is None:
        screenshot_times = []
    if screenshot_dir is None:
        screenshot_dir = os.path.join(os.path.dirname(output_path), 'screenshots')

    if x_range is None:
        x_range = (-3.5, 3.5)
    xs = torch.linspace(*x_range, 1000)
    if y_range is None:
        y_range = (-3.5, 3.5)
    ys = torch.linspace(*y_range, 1000)

    # Create meshgrid for the surface plot
    x, y = np.meshgrid(xs, ys)
    z_input = torch.from_numpy(np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1))
    z = function(z_input).reshape(1000, 1000).numpy()
    images = []

    snapshot_data = []

    for i, trajectory_point in enumerate(trajectory):
        V, V_alpha, V_best = trajectory_point['V'], trajectory_point['V_alpha'], trajectory_point['V_best']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface with a colormap
        surface = ax.plot_surface(x, y, z, cmap='magma', alpha=0.6)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

        # Set viewing angles
        ax.view_init(elev=elev_deg if elev_deg is not None else 30,
                     azim=azim_deg if azim_deg is not None else 30)

        # Remove z-axis ticks and label
        ax.set_zticks([])
        ax.set_zlabel('')

        # Scatter plot of particles, best particle, and consensus point
        ax.scatter(V[:, 0], V[:, 1], function(V), marker='o', color='black', label='particles', zorder=2)
        ax.scatter(V_best[0], V_best[1], function(V_best.reshape(1, -1)), marker='o', color='black', label='best', zorder=3)
        ax.scatter(V_alpha[0], V_alpha[1], function(V_alpha.reshape(1, -1)), marker='o', color='black', label='consensus', zorder=4)

        # Show legend
        # ax.legend()

        plt.title(f'time = {(dt * i):.2f}, consensus = {V_alpha[0].item():.2f},{V_alpha[1].item():.2f}')

        # Save temporary image for GIF creation
        tmp_path = os.path.join(os.path.dirname(output_path), f'{i}_tmp.png')
        plt.savefig(tmp_path, bbox_inches='tight')
        images.append(iio.imread(tmp_path))
        os.remove(tmp_path)

        current_time = round(i * dt, 8)

        # Collect data for requested screenshot times
        if any(abs(current_time - t) < 1e-8 for t in screenshot_times):
            snapshot_data.append({
                'V': V,
                'V_alpha': V_alpha,
                'V_best': V_best,
                'time': current_time
            })

        plt.close(fig)

        # Save the GIF animation
    if os.path.exists(output_path):
        os.remove(output_path)
    imageio.mimsave(output_path, images)

    # Save summary figure with all requested snapshots in 2x2 grid
    if snapshot_data:
        import math

        n_snapshots = len(snapshot_data)
        
        # --------------- Variante 1: 2x2-Grid ---------------
        n_cols_grid = 2
        n_rows_grid = math.ceil(n_snapshots / n_cols_grid)

        fig_grid, axes_grid = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid),
                                           subplot_kw={'projection': '3d'})
        if n_rows_grid == 1 and n_cols_grid == 1:
            axes_grid = [axes_grid]
        else:
            axes_grid = axes_grid.flatten()

        for ax in axes_grid[n_snapshots:]:
            ax.axis('off')

        for ax, snapshot in zip(axes_grid, snapshot_data):
            V, V_alpha, V_best = snapshot['V'], snapshot['V_alpha'], snapshot['V_best']
            time = snapshot['time']

            surface = ax.plot_surface(x, y, z, cmap='magma', alpha=0.6)
            ax.view_init(elev=elev_deg if elev_deg is not None else 30,
                         azim=azim_deg if azim_deg is not None else 30)
            ax.set_zticks([])
            ax.set_zlabel('')
            ax.set_title(f'time = {time:.2f}, consensus = {V_alpha[0].item():.2f},{V_alpha[1].item():.2f}')
            ax.scatter(V[:, 0], V[:, 1], function(V), marker='o', color='black')
            ax.scatter(V_best[0], V_best[1], function(V_best.reshape(1, -1)), marker='o', color='black')
            ax.scatter(V_alpha[0], V_alpha[1], function(V_alpha.reshape(1, -1)), marker='o', color='black')

        if summary_title is not None:
            fig_grid.suptitle(summary_title, fontsize=16, y=1.02)

        fig_grid.subplots_adjust(right=0.85, left=0.05, bottom=0.05, top=0.9)
        cbar_ax_grid = fig_grid.add_axes([0.88, 0.15, 0.02, 0.7])
        fig_grid.colorbar(surface, cax=cbar_ax_grid)

        path_grid = screenshot_dir + '_grid.png'
        fig_grid.savefig(path_grid, bbox_inches='tight')
        plt.close(fig_grid)

        # --------------- Variante 2: alle vier nebeneinander ---------------
        if n_snapshots == 4:
            fig_row, axes_row = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})

            for ax, snapshot in zip(axes_row, snapshot_data):
                V, V_alpha, V_best = snapshot['V'], snapshot['V_alpha'], snapshot['V_best']
                time = snapshot['time']

                surface = ax.plot_surface(x, y, z, cmap='magma', alpha=0.6)
                ax.view_init(elev=elev_deg if elev_deg is not None else 30,
                             azim=azim_deg if azim_deg is not None else 30)
                ax.set_zticks([])
                ax.set_zlabel('')
                ax.set_title(f'time = {time:.2f}, consensus = {V_alpha[0].item():.2f},{V_alpha[1].item():.2f}')
                ax.scatter(V[:, 0], V[:, 1], function(V), marker='o', color='black')
                ax.scatter(V_best[0], V_best[1], function(V_best.reshape(1, -1)), marker='o', color='black')
                ax.scatter(V_alpha[0], V_alpha[1], function(V_alpha.reshape(1, -1)), marker='o', color='black')

            if summary_title is not None:
                fig_row.suptitle(summary_title, fontsize=16, y=1.05)

            fig_row.subplots_adjust(right=0.85, left=0.05, bottom=0.05, top=0.9)
            cbar_ax_row = fig_row.add_axes([0.88, 0.15, 0.02, 0.7])
            fig_row.colorbar(surface, cax=cbar_ax_row)

            path_row = screenshot_dir + '_row.png'
            fig_row.savefig(path_row, bbox_inches='tight')
            plt.close(fig_row)











# visualization of convergence rate
def visualize_trajectory_convergence(trajectory, minimizer, display_exponent=False, l=1., sigma=1., dt=0.01,
                                     output_path=None):
    plt.rcParams.update({'figure.figsize': (6, 6)})
    convergence_measure = lambda points: (torch.norm(points - minimizer, p=2, dim=1) ** 2).sum().detach().numpy() / 2
    timestamps = dt * np.arange(len(trajectory))
    values = np.array([convergence_measure(t['V']) for t in trajectory])
    plt.clf()
    plt.plot(timestamps, values / values[0], label='result')
    if display_exponent:
        plt.plot(timestamps, np.exp(-timestamps * (2 * l - sigma ** 2)),
                 label=r'$e^{-(2\lambda - \sigma^2)t}$', linestyle='--', alpha=0.5)
        plt.legend(prop={'size': 18})
    plt.xlabel('t', fontsize=18)
    plt.ylabel(r'$\frac{V^N(\rho_t^N)}{V^N(\rho_0^N)}$', fontsize=18)
    #if output_path is not None:
    #    plt.savefig(output_path)
    #plt.show()




# visualization of several particles (number_of_particles) in one dimension (dim) and their trajectory (to see consensus forming)
def visualize_trajectory_particles(trajectory, dt, vector, particles, dim, output_path=None):
# INPUT: - trajectory (dictionary): stores all agent positions (epoch x N x d), all CP (epoch x 1 x d) and all best agents (epoch x 1 x d)
#        - dt (scalar): step size of discretization
#        - vector (string): which particle shall be plottet from {'V',V_best','V_alpha'}
#        - particles (list): set of particles indices in [1,...,N] or [1] if vector is 'V_best' or 'V_alpha'
#        - dim (scalar): dimension we want to extract in {1,...,dimensionality}
#        - output_path (string): direction where to store the the plot (is not stored if None)

    idx = [x - 1 for x in particles] # get particles
    if vector == 'V':
        first_coordinates_list = [trajectory_point[vector][idx, dim-1] for trajectory_point in trajectory] # get particles position
    else:
        first_coordinates_list = [trajectory_point[vector][dim-1] for trajectory_point in trajectory] # get particles position 

    coords = [[] for _ in range(len(idx))] # empty list for all coordinates

    # extract all coordiantes of 
    for j in range(0, len(first_coordinates_list), 3):
        t = first_coordinates_list[j]
        for i in range(len(idx)):  # for each coordiante in the tensor
            if vector == 'V':
                coords[i].append(t[i].item())
            else:
                coords[i].append(t.item())

    plt.figure(figsize=(7, 5)) # plot

    if vector == 'V':
        for i in range(len(idx)):
            plt.plot([dt * j for j in range(0, len(first_coordinates_list), 3)],[x for x in coords[i]], marker='o', label=f'Koordinate {i+1}') # plot each coordinate
    else:
        for i in range(len(idx)):
            plt.plot([dt * j for j in range(0, len(first_coordinates_list), 3)],[x for x in coords[i]], marker='o') # plot each coordinate

    plt.title(f'trajectory of coordinates N= {particles} of {vector} in dimension {dim}')
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend()

    # save plot
    plt.tight_layout()  # Stellt sicher, dass die Labels nicht abgeschnitten werden
    if output_path is not None:
        plt.savefig(output_path)




# visualization of the evolution/trajectory of several particles (number_of_particles) in one dimension (dim) (to see how they develope)
def visualize_trajectory_particles_evolution(trajectory, dt, vector, particles, dim, output_path=None):
# INPUT: - trajectory (dictionary): stores all agent positions (epoch x N x d), all CP (epoch x 1 x d) and all best agents (epoch x 1 x d)
#        - dt (scalar): step size of discretization
#        - vector (string): which particle shall be plottet from {'V',V_best','V_alpha'}
#        - particles (list): set of particles indices in [1,...,N] or [1] if vector is 'V_best' or 'V_alpha'
#        - dim (scalar): dimension we want to extract in {1,...,dimensionality}
#        - output_path (string): direction where to store the the plot (is not stored if None)

    idx = [x - 1 for x in particles] # get particles
    if vector == 'V':
        first_coordinates_list = [trajectory_point[vector][idx, dim-1] for trajectory_point in trajectory] # get particles position
    else:
        first_coordinates_list = [trajectory_point[vector][dim-1] for trajectory_point in trajectory] # get particles position 

    coords = [[] for _ in range(len(idx))] # empty list for all coordinates

    # extract all coordiantes of 
    for j in range(1, len(first_coordinates_list), 3):
        s = first_coordinates_list[j-1]
        t = first_coordinates_list[j]
        for i in range(len(idx)):  # for each coordiante in the tensor
            if vector == 'V':
                coords[i].append(t[i].item()-s[i].item())
            else:
                coords[i].append(t.item()-s.item())

    plt.figure(figsize=(7, 5)) # plot

    if vector == 'V':
        for i in range(len(idx)):
            plt.plot([dt * j for j in range(1, len(first_coordinates_list), 3)],[x for x in coords[i]], marker='o', label=f'Koordinate {i+1}') # plot each coordinate
    else:
        for i in range(len(idx)):
            plt.plot([dt * j for j in range(1, len(first_coordinates_list), 3)],[x for x in coords[i]], marker='o') # plot each coordinate

    plt.title(f'trajectory of evolution of coordinates N={particles} of {vector} in dimension {dim}')
    plt.xlabel('time')
    plt.ylabel('evolution')
    plt.legend()

    # save plot
    plt.tight_layout()  # Stellt sicher, dass die Labels nicht abgeschnitten werden
    if output_path is not None:
        plt.savefig(output_path)





# visualization the behaviour of the difference between two particles (number_of_particles) in one dimension (dim) (to see consensus forming)
def visualize_trajectory_particle_differences(trajectory, dt, particles, dim, output_path=None):
# INPUT: - trajectory (dictionary): stores all agent positions (epoch x N x d), all CP (epoch x 1 x d) and all best agents (epoch x 1 x d)
#        - dt (scalar): step size of discretization
#        - vector (string): which particle shall be plottet from {'V',V_best','V_alpha'}
#        - particles (list): set of particles indices in [1,...,N] or [1] if vector is 'V_best' or 'V_alpha'
#        - dim (scalar): dimension we want to extract in {1,...,dimensionality}
#        - output_path (string): direction where to store the the plot (is not stored if None)

    idx = [x - 1 for x in particles] # get particles
    first_coordinates_list = [trajectory_point['V'][idx, dim-1] for trajectory_point in trajectory] # get particles position

    coords = [[] for _ in range(int(len(idx)/2))] # empty list for all coordinates

    # extract all coordiantes of 
    for j in range(0, len(first_coordinates_list), 3):
        t = first_coordinates_list[j]
        for i in range(1,len(idx),2):  # for each coordiante in the tensor
            coords[int(i/2)].append(abs(t[i].item()-t[i-1].item()))

    plt.figure(figsize=(7, 5)) # plot

    for i in range(int(len(idx)/2)):
        plt.plot([dt * j for j in range(0, len(first_coordinates_list), 3)],[x for x in coords[i]], marker='o', label=f'Koordinate {i+1} und {i+2}') # plot each coordinate

    plt.title(f'trajectory of the differences between coordinates N={particles} in dimension {dim}')
    plt.xlabel('time')
    plt.ylabel('diffrences')
    plt.legend()

    # save plot
    plt.tight_layout()  # Stellt sicher, dass die Labels nicht abgeschnitten werden
    if output_path is not None:
        plt.savefig(output_path)
   
