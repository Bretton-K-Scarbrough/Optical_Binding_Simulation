"""
Plot and animate saved optical binding simulation data.

This script loads saved NumPy arrays from the simulation output directory and
provides plotting utilities for particle positions, velocities, forces, fields,
dipole moments, and Green's function diagnostics. It can also generate 3D
particle trajectory frames and compile them into an MP4 video.
"""

import numpy as np
import matplotlib.pyplot as plt
from constants import num_of_particle, L
from tqdm.auto import tqdm
import os
import glob
import re
import imageio.v2 as imageio


path = "./data/"
frame_time_step = 1000
delete_frames = False
data = np.load(path + "position_data.npy")
length = data.shape[0] - 1


def position_data_3D():
    """
    Display the particle positions at a single simulation time step in 3D.

    This function loads the saved position data, extracts the particle positions
    at the selected time step, and displays them as a 3D scatter plot.
    """
    data = np.load(path + "position_data.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    maxstep = 0
    for n in range(num_of_particle):
        x = data[(maxstep), n, 0]
        y = data[(maxstep), n, 1]
        z = data[(maxstep), n, 2]

        ax.scatter(x, y, z)
    plt.show()


def grad_force_data():
    """
    Plot the gradient force components over time for each particle.

    This function loads the saved gradient force data and creates separate
    time-series plots for the x, y, and z force components. The resulting figure
    is saved to disk and displayed.
    """
    data = np.load(path + "grad_force_data.npy")
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].set_title("x Grad Force vs Time")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Force")
    axs[0].grid(True)

    for i in range(num_of_particle):
        axs[0].plot(data[0:length, i, 0])

    axs[1].set_title("y Grad Force vs Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force")
    axs[1].grid(True)

    for i in range(num_of_particle):
        axs[1].plot(data[0:length, i, 1])

    axs[2].set_title("z Grad Force vs Time")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Force")
    axs[2].grid(True)

    for i in range(num_of_particle):
        axs[2].plot(data[0:length, i, 2])

    plt.tight_layout()
    plt.savefig("./data/grad_force_data.png", dpi=300)
    plt.show()


def rad_force_data():
    """
    Plot the radiation pressure force components over time for each particle.

    This function loads the saved radiation pressure force data and creates
    separate time-series plots for the x, y, and z force components. The
    resulting figure is saved to disk and displayed.
    """
    data = np.load(path + "rad_force_data.npy")
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].set_title("x Rad Force vs Time")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Force")
    axs[0].grid(True)

    for i in range(num_of_particle):
        axs[0].plot(data[0:length, i, 0])

    axs[1].set_title("y Rad Force vs Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force")
    axs[1].grid(True)

    for i in range(num_of_particle):
        axs[1].plot(data[0:length, i, 1])

    axs[2].set_title("z Rad Force vs Time")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Force")
    axs[2].grid(True)

    for i in range(num_of_particle):
        axs[2].plot(data[0:length, i, 2])

    plt.tight_layout()
    plt.savefig("./data/rad_force_data.png", dpi=300)
    plt.show()


def spin_force_data():
    """
    Plot the spin-force components over time for each particle.

    This function loads the saved spin-force data and creates separate
    time-series plots for the x, y, and z force components. The resulting figure
    is saved to disk and displayed.
    """
    data = np.load(path + "spin_force_data.npy")
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].set_title("x Spin Force vs Time")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Force")
    axs[0].grid(True)

    for i in range(num_of_particle):
        axs[0].plot(data[0:length, i, 0])

    axs[1].set_title("y Spin Force vs Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force")
    axs[1].grid(True)

    for i in range(num_of_particle):
        axs[1].plot(data[0:length, i, 1])

    axs[2].set_title("z Spin Force vs Time")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Force")
    axs[2].grid(True)

    for i in range(num_of_particle):
        axs[2].plot(data[0:length, i, 2])

    plt.tight_layout()
    plt.savefig("./data/spin_force_data.png", dpi=300)
    plt.show()


def gen_frames():
    """
    Generate 3D animation frames from saved particle position data.

    This function loads the saved position history, plots the particle positions
    at selected time steps, rotates the 3D camera view over time, and saves each
    rendered frame as an image file.
    """
    data = np.load(path + "position_data.npy")

    time_steps = data.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    azim_0 = 45
    elev_0 = 45
    elev = elev_0
    deg_per_frame = 0.5

    s_near = 60
    s_far = 10

    for frame_number, t in enumerate(
        tqdm(
            range(0, time_steps, frame_time_step), desc="Generating frames", unit="step"
        )
    ):
        ax.cla()

        # Get all particles at this frame
        x = data[t, :, 0]
        y = data[t, :, 1]
        z = data[t, :, 2]

        azim = azim_0 + deg_per_frame * frame_number
        # elev = elev_0 + deg_per_frame * frame_number
        ax.view_init(elev=elev, azim=azim)  # type: ignore

        elev_rad = np.deg2rad(elev)
        azim_rad = np.deg2rad(azim)

        elev_rad = 0
        azim_rad = np.pi / 2

        # Approximate viewing direction
        view_dir = np.array(
            [
                np.cos(elev_rad) * np.cos(azim_rad),
                np.cos(elev_rad) * np.sin(azim_rad),
                np.sin(elev_rad),
            ]
        )

        # Shape: (num_particles, 3)
        pts = np.column_stack((x, y, z))

        # One depth value per particle
        depth = pts @ view_dir

        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Nearer points bigger
        sizes = s_far + (1.0 - depth_norm) * (s_near - s_far)

        # ax.scatter(x, y, z, s=sizes, depthshade=True)
        ax.scatter(x, y, z, depthshade=True)

        ax.set_xlim(-L / 2, L / 2)
        ax.set_ylim(-L / 2, L / 2)
        z_pad = 0.1 * L
        ax.set_zlim(np.min(z) - z_pad, np.max(z) + z_pad)  # type: ignore

        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")  # type: ignore

        plt.savefig(f"./plotting/videos/frames/3D/time_{t}.png", dpi=200)


def gen_video():
    """
    Compile saved 3D animation frames into an MP4 video.

    This function loads the generated frame images, sorts them by simulation time
    step, writes them to a video file, and optionally deletes the intermediate
    frame images after compilation.
    """
    # images = glob.glob("./data/frames/2D/time_*.png")
    images = glob.glob("./plotting/videos/frames/3D/time_*.png")
    images.sort(key=lambda f: int(re.search(r"time_(\d+)", f).group(1)))  # type: ignore

    # with imageio.get_writer("./data/video_2D.mp4", fps=60) as writer:
    with imageio.get_writer("./plotting/videos/video_3D.mp4", fps=30) as writer:
        for filename in tqdm(images, desc="Compiling frames", unit="frame"):
            image = imageio.imread(filename)
            writer.append_data(image)  # type: ignore

    print("Generated video")

    if delete_frames:
        for image in images:
            try:
                os.remove(image)
            except OSError as e:
                print(f"Error deleting {image}: {e}")


def position_data():
    """
    Plot the x, y, and z particle positions over time.

    This function loads the saved position data and creates separate time-series
    plots for each position component for all particles.
    """
    data = np.load(path + "position_data.npy")
    plt.figure()
    plt.title("x Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.savefig(path + "x Position vs Time.png")

    plt.figure()
    plt.title("y Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Position vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def velocity_data():
    """
    Plot the x, y, and z particle velocities over time.

    This function loads the saved velocity data and creates separate time-series
    plots for each velocity component for all particles.
    """
    data = np.load(path + "velocity_data.npy")
    plt.figure()
    plt.title("x Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.savefig(path + "x Velocity vs Time.png")

    plt.figure()
    plt.title("y Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Velocity vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def force_data():
    """
    Plot the total force components over time for each particle.

    This function loads the saved total force data and creates separate
    time-series plots for the x, y, and z force components.
    """
    data = np.load(path + "forces_data.npy")
    plt.figure()
    plt.title("x Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Force vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def E_field_data():
    """
    Plot the electric field components over time at each particle position.

    This function loads the saved electric field data and creates separate
    time-series plots for the x, y, and z electric field components.
    """
    data = np.load(path + "E_n_data.npy")
    plt.figure()
    plt.title("x E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z E_field_strength vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("E_field_strength")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def dipole_moment_data():
    """
    Plot the induced dipole moment components over time for each particle.

    This function loads the saved dipole moment data and creates separate
    time-series plots for the x, y, and z dipole moment components.
    """
    data = np.load(path + "p_i_data.npy")
    plt.figure()
    plt.title("x Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 0])

    plt.figure()
    plt.title("y Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 1])

    plt.figure()
    plt.title("z Dipole Moment vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dipole Moment")
    plt.grid(True)

    for i in range(num_of_particle):
        plt.plot(data[0:length, i, 2])

    plt.show()


def G_max_det():
    """
    Plot Green's function matrix diagnostics over time.

    This function loads the saved determinant and maximum-value data for the
    flattened Green's function matrix and displays them as time-series plots.
    """
    det_data = np.load(path + "G_det.npy")
    max_data = np.load(path + "G_max.npy")

    plt.figure()
    plt.title("G Det Data vs Time")
    plt.grid(True)
    plt.plot(det_data)

    plt.figure()
    plt.title("G Max Data vs Time")
    plt.grid(True)
    plt.plot(max_data)

    plt.show()


if __name__ == "__main__":
    # position_data_3D()
    # gen_frames()
    # gen_video()
    position_data()
    # velocity_data()
    # grad_force_data()
    # rad_force_data()
    # spin_force_data()
    # force_data()
    # E_field_data()
    # dipole_moment_data()
    # G_max_det()
