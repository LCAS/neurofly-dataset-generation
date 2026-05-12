from neurofly_dataset_generation.cli import run_noise_cli
from neurofly_dataset_generation.noise import generate_noisy_csv

__all__ = ["generate_noisy_csv"]


if __name__ == "__main__":
    run_noise_cli()
