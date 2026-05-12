"""Plotting utilities for generated and noisy datasets."""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np


POSITION_COLUMNS = ("x", "y", "z")
NOISY_POSITION_COLUMNS = ("mocap_x_noisy", "mocap_y_noisy", "mocap_z_noisy")


def _results_to_arrays(results):
    return {
        "time": np.asarray(results["time"]),
        "position": np.asarray(results["state"]["x"]),
        "desired_position": np.asarray(results["flat"]["x"]),
    }


def _csv_to_arrays(csv_path):
    time = []
    position = []
    desired_position = []

    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            time.append(float(row["time"]))
            position.append([float(row["x"]), float(row["y"]), float(row["z"])])
            desired_position.append(
                [float(row["xdes"]), float(row["ydes"]), float(row["zdes"])]
            )

    return {
        "time": np.asarray(time),
        "position": np.asarray(position),
        "desired_position": np.asarray(desired_position),
    }


def load_trajectory_data(results=None, csv_path="basic_usage.csv"):
    if results is not None:
        return _results_to_arrays(results)
    return _csv_to_arrays(csv_path)


def plot_trajectory_comparison(
    results=None,
    csv_path="basic_usage.csv",
    show=True,
    output_dir=None,
    file_prefix="trajectory_comparison",
):
    data = load_trajectory_data(results=results, csv_path=csv_path)
    time = data["time"]
    position = data["position"]
    desired_position = data["desired_position"]
    position_error = np.linalg.norm(position - desired_position, axis=1)

    fig_3d = plt.figure("Desired vs Followed Trajectory", figsize=(9, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(
        desired_position[:, 0],
        desired_position[:, 1],
        desired_position[:, 2],
        color="black",
        linewidth=2,
        label="Desired trajectory",
    )
    ax_3d.plot(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        color="tab:blue",
        linewidth=1.8,
        label="Followed trajectory",
    )
    ax_3d.scatter(
        desired_position[0, 0],
        desired_position[0, 1],
        desired_position[0, 2],
        color="tab:green",
        s=40,
        label="Start",
    )
    ax_3d.scatter(
        desired_position[-1, 0],
        desired_position[-1, 1],
        desired_position[-1, 2],
        color="tab:red",
        s=40,
        label="End",
    )
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("Desired and Followed UAV Trajectories")
    ax_3d.legend()

    fig_time, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(11, 10),
        sharex=True,
        num="Trajectory Tracking vs Time",
    )
    for axis_index, (axis_label, color) in enumerate(
        zip(("X", "Y", "Z"), ("tab:red", "tab:green", "tab:blue"))
    ):
        axes[axis_index].plot(
            time,
            desired_position[:, axis_index],
            color=color,
            linewidth=2,
            label=f"{axis_label} desired",
        )
        axes[axis_index].plot(
            time,
            position[:, axis_index],
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{axis_label} followed",
        )
        axes[axis_index].set_ylabel(f"{axis_label} (m)")
        axes[axis_index].grid(True)
        axes[axis_index].legend(loc="upper right")

    axes[3].plot(time, position_error, color="tab:purple", linewidth=1.8)
    axes[3].set_ylabel("Error (m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Position Tracking Error")
    axes[3].grid(True)
    fig_time.tight_layout()

    print(
        "Trajectory comparison metrics:",
        {
            "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
            "position_max_m": float(np.max(position_error)),
        },
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_3d.savefig(output_dir / f"{file_prefix}_3d.png", bbox_inches="tight", dpi=200)
        fig_time.savefig(
            output_dir / f"{file_prefix}_time.png",
            bbox_inches="tight",
            dpi=200,
        )

    if show:
        plt.show()

    return fig_3d, fig_time


def load_position_data(csv_path):
    time = []
    clean_position = []
    noisy_position = []

    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = ("time", *POSITION_COLUMNS, *NOISY_POSITION_COLUMNS)
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required columns for plotting: {', '.join(missing)}"
            )

        for row in reader:
            time.append(float(row["time"]))
            clean_position.append([float(row[column]) for column in POSITION_COLUMNS])
            noisy_position.append(
                [float(row[column]) for column in NOISY_POSITION_COLUMNS]
            )

    return (
        np.asarray(time),
        np.asarray(clean_position),
        np.asarray(noisy_position),
    )


def plot_noisy_vs_clean_position(csv_path, output_dir, prefix, show=False):
    time, clean_position, noisy_position = load_position_data(csv_path)
    position_error = noisy_position - clean_position
    error_norm = np.linalg.norm(position_error, axis=1)

    fig_3d = plt.figure("Clean vs Noisy Position", figsize=(9, 8))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot(
        clean_position[:, 0],
        clean_position[:, 1],
        clean_position[:, 2],
        color="black",
        linewidth=2,
        label="Clean position",
    )
    ax_3d.plot(
        noisy_position[:, 0],
        noisy_position[:, 1],
        noisy_position[:, 2],
        color="tab:orange",
        linewidth=1.5,
        alpha=0.9,
        label="Noisy position",
    )
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("UAV Position: Clean vs Noisy")
    ax_3d.legend()

    fig_time, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(11, 10),
        sharex=True,
        num="Clean vs Noisy Position vs Time",
    )
    for axis_index, (axis_label, color) in enumerate(
        zip(("X", "Y", "Z"), ("tab:red", "tab:green", "tab:blue"))
    ):
        axes[axis_index].plot(
            time,
            clean_position[:, axis_index],
            color=color,
            linewidth=2,
            label=f"{axis_label} clean",
        )
        axes[axis_index].plot(
            time,
            noisy_position[:, axis_index],
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label=f"{axis_label} noisy",
        )
        axes[axis_index].set_ylabel(f"{axis_label} (m)")
        axes[axis_index].grid(True)
        axes[axis_index].legend(loc="upper right")

    axes[3].plot(time, error_norm, color="tab:purple", linewidth=1.8)
    axes[3].set_ylabel("Error (m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Noisy Position Error Norm")
    axes[3].grid(True)
    fig_time.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_3d.savefig(output_dir / f"{prefix}_3d.png", bbox_inches="tight", dpi=200)
    fig_time.savefig(output_dir / f"{prefix}_time.png", bbox_inches="tight", dpi=200)

    print(
        "Noisy position metrics:",
        {
            "rmse_m": float(np.sqrt(np.mean(error_norm**2))),
            "max_error_m": float(np.max(error_norm)),
        },
    )

    if show:
        plt.show()

    return fig_3d, fig_time
