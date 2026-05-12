# neurofly-dataset-generation

Utilities for generating UAV trajectory datasets in RotorPy, adding configurable sensor noise, and plotting the resulting trajectories. The current workflow focuses on a smooth conical spiral flight path and produces CSV files that are useful for filtering, state-estimation, and sensor-fusion experiments.

## Purpose

This repository provides three main capabilities:

1. Run a RotorPy simulation of a spiral flight path and export the resulting CSV logs.
2. Augment a simulation CSV with noisy IMU, mocap or GPS-style measurements, plus a drifting dead-reckoned state.
3. Visualize tracking quality and noisy-vs-clean position data.

The code is organized as a small Python package in [`neurofly_dataset_generation`](./neurofly_dataset_generation), while the original top-level scripts remain as compatibility entry points:

- [`main.py`](./main.py): run the simulation and export datasets
- [`inject_trajectory_noise.py`](./inject_trajectory_noise.py): create a noisy CSV
- [`visualise_traj.py`](./visualise_traj.py): plot desired vs followed trajectory
- [`plot_noisy_position.py`](./plot_noisy_position.py): plot clean vs noisy position

The preferred interface is now the package CLI:

```bash
python -m neurofly_dataset_generation <command> [options]
```

Available commands:

- `generate`
- `add-noise`
- `plot-tracking`
- `plot-noisy`

## Dependencies

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`
- `rotorpy`

Dependencies are declared in [`pyproject.toml`](./pyproject.toml) and [`requirements.txt`](./requirements.txt).

## Setup

If you already have a working environment with RotorPy installed, you can use it directly. In this workspace there is also a local virtual environment under `rotorpy/`.

Example setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you prefer the existing local environment in this folder:

```bash
rotorpy/bin/pip install -e .
```

## Usage

### 1. Generate a simulation dataset

```bash
rotorpy/bin/python main.py --plot-tracking --tracking-output-dir trajectory_plots
```

Equivalent package CLI:

```bash
rotorpy/bin/python -m neurofly_dataset_generation generate --plot-tracking --tracking-output-dir trajectory_plots
```

This writes:

- `basic_usage.csv`: full RotorPy state and reference log
- `spiral_flight_data.csv`: compact IMU-oriented export
- `trajectory_plots/`: optional tracking plots

Useful flags:

- `--no-plot`
- `--no-animate`
- `--quiet`
- `--sensor-csv <path>`
- `--simulation-csv <path>`

### 2. Inject sensor noise

```bash
python3 inject_trajectory_noise.py basic_usage.csv --output basic_usage_noisy.csv --abs-sensor mocap --seed 7
```

Equivalent package CLI:

```bash
python3 -m neurofly_dataset_generation add-noise basic_usage.csv --output basic_usage_noisy.csv --abs-sensor mocap --seed 7
```

Options:

- `--abs-sensor mocap`
- `--abs-sensor gps`
- `--abs-sensor none`
- `--noise-scale <float>`
- `--mode deadreckon`

The generated noisy CSV appends:

- noisy accelerometer and gyro channels
- optional noisy mocap or GPS-like position channels
- noisy wind and rotor-speed channels
- a dead-reckoned drifting state estimate

### 3. Plot desired vs tracked trajectory

```bash
python3 visualise_traj.py basic_usage.csv --output-dir trajectory_plots --show
```

Equivalent package CLI:

```bash
python3 -m neurofly_dataset_generation plot-tracking basic_usage.csv --output-dir trajectory_plots --show
```

### 4. Plot clean vs noisy position

```bash
python3 plot_noisy_position.py basic_usage_noisy.csv --output-dir trajectory_plots --show
```

Equivalent package CLI:

```bash
python3 -m neurofly_dataset_generation plot-noisy basic_usage_noisy.csv --output-dir trajectory_plots --show
```

## Project Structure

```text
neurofly_dataset_generation/
  cli.py           Command-line entry points
  config.py        Shared defaults
  io_utils.py      CSV and filesystem helpers
  noise.py         Noise injection and dead reckoning
  plotting.py      Visualization helpers
  simulation.py    RotorPy simulation orchestration
  trajectories.py  Spiral trajectory definitions
```

## Notes

- Generated CSV files and plot outputs are ignored by Git by default.
- The local `rotorpy/` directory is treated as an environment directory and is also ignored.
- The simulation entry point requires a Python environment where the `rotorpy` package is installed.
