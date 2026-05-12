from neurofly_dataset_generation.cli import run_tracking_plot_cli
from neurofly_dataset_generation.plotting import (
    load_trajectory_data,
    plot_trajectory_comparison,
)

__all__ = ["load_trajectory_data", "plot_trajectory_comparison"]


if __name__ == "__main__":
    run_tracking_plot_cli()
