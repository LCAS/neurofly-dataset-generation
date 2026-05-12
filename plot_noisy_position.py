from neurofly_dataset_generation.cli import run_noisy_plot_cli
from neurofly_dataset_generation.plotting import (
    load_position_data,
    plot_noisy_vs_clean_position,
)

__all__ = ["load_position_data", "plot_noisy_vs_clean_position"]


if __name__ == "__main__":
    run_noisy_plot_cli()
