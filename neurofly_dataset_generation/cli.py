"""Command-line entry points for dataset generation workflows."""

import argparse

from .config import (
    DEFAULT_NOISY_CSV,
    DEFAULT_PLOT_DIR,
    DEFAULT_SENSOR_CSV,
    DEFAULT_SIM_CSV,
)


def build_dataset_parser():
    parser = argparse.ArgumentParser(
        description="Run the RotorPy spiral simulation and export dataset CSV files."
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable RotorPy plots.")
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Disable RotorPy animation output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument(
        "--plot-tracking",
        action="store_true",
        help="Create followed-vs-desired trajectory figures.",
    )
    parser.add_argument(
        "--tracking-output-dir",
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for tracking plots. Defaults to {DEFAULT_PLOT_DIR}.",
    )
    parser.add_argument(
        "--sensor-csv",
        default=DEFAULT_SENSOR_CSV,
        help=f"Output path for the IMU-only CSV export. Defaults to {DEFAULT_SENSOR_CSV}.",
    )
    parser.add_argument(
        "--simulation-csv",
        default=DEFAULT_SIM_CSV,
        help=f"Output path for RotorPy's full CSV export. Defaults to {DEFAULT_SIM_CSV}.",
    )
    return parser


def build_noise_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Append noisy sensor channels and a drifting dead-reckoned state to a "
            "RotorPy CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=DEFAULT_SIM_CSV,
        help=f"Source CSV path. Defaults to {DEFAULT_SIM_CSV}.",
    )
    parser.add_argument(
        "--output",
        dest="output_csv",
        help="Optional output CSV path. Defaults to <input>_noisy.csv.",
    )
    parser.add_argument(
        "--mode",
        choices=("deadreckon",),
        default="deadreckon",
        help="State propagation mode. Defaults to deadreckon.",
    )
    parser.add_argument(
        "--abs-sensor",
        choices=("mocap", "gps", "none"),
        default="mocap",
        help="Absolute position sensor model to append. Defaults to mocap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible noise generation. Defaults to 7.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Global multiplier for injected noise and bias sigmas.",
    )
    return parser


def build_tracking_plot_parser():
    parser = argparse.ArgumentParser(
        description="Plot desired vs followed trajectory from a RotorPy CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_SIM_CSV,
        help=f"Path to the simulation CSV. Defaults to {DEFAULT_SIM_CSV}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for saved plots. Defaults to {DEFAULT_PLOT_DIR}.",
    )
    parser.add_argument(
        "--prefix",
        default="trajectory_comparison",
        help="Filename prefix for saved plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser


def build_noisy_plot_parser():
    parser = argparse.ArgumentParser(
        description="Plot clean and noisy UAV position from a noisy RotorPy CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_NOISY_CSV,
        help=f"Path to the noisy CSV. Defaults to {DEFAULT_NOISY_CSV}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for saved plots. Defaults to {DEFAULT_PLOT_DIR}.",
    )
    parser.add_argument(
        "--prefix",
        default="noisy_vs_clean_position",
        help="Filename prefix for saved plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser


def run_dataset_cli(argv=None):
    from .simulation import run_spiral_study

    args = build_dataset_parser().parse_args(argv)
    run_spiral_study(
        plot=not args.no_plot,
        animate=not args.no_animate,
        verbose=not args.quiet,
        plot_tracking=args.plot_tracking,
        tracking_output_dir=args.tracking_output_dir,
        sensor_csv_path=args.sensor_csv,
        simulation_csv_path=args.simulation_csv,
    )


def run_noise_cli(argv=None):
    from .noise import generate_noisy_csv

    args = build_noise_parser().parse_args(argv)
    written_path = generate_noisy_csv(
        input_path=args.input_csv,
        output_path=args.output_csv,
        mode=args.mode,
        abs_sensor=args.abs_sensor,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    print(f"Wrote noisy trajectory CSV to {written_path}")


def run_tracking_plot_cli(argv=None):
    from .plotting import plot_trajectory_comparison

    args = build_tracking_plot_parser().parse_args(argv)
    plot_trajectory_comparison(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        file_prefix=args.prefix,
        show=args.show,
    )


def run_noisy_plot_cli(argv=None):
    from .plotting import plot_noisy_vs_clean_position

    args = build_noisy_plot_parser().parse_args(argv)
    plot_noisy_vs_clean_position(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
        show=args.show,
    )


def build_root_parser():
    parser = argparse.ArgumentParser(
        description="Unified CLI for RotorPy dataset generation utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "generate",
        parents=[build_dataset_parser()],
        add_help=False,
        help="Run the RotorPy spiral simulation and export dataset CSV files.",
    )
    subparsers.add_parser(
        "add-noise",
        parents=[build_noise_parser()],
        add_help=False,
        help="Append noisy sensor channels to a RotorPy CSV.",
    )
    subparsers.add_parser(
        "plot-tracking",
        parents=[build_tracking_plot_parser()],
        add_help=False,
        help="Plot desired vs followed trajectory from a simulation CSV.",
    )
    subparsers.add_parser(
        "plot-noisy",
        parents=[build_noisy_plot_parser()],
        add_help=False,
        help="Plot clean vs noisy position from a noisy CSV.",
    )
    return parser


def main(argv=None):
    parser = build_root_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        run_dataset_cli(argv[1:] if argv is not None else None)
        return
    if args.command == "add-noise":
        run_noise_cli(argv[1:] if argv is not None else None)
        return
    if args.command == "plot-tracking":
        run_tracking_plot_cli(argv[1:] if argv is not None else None)
        return
    if args.command == "plot-noisy":
        run_noisy_plot_cli(argv[1:] if argv is not None else None)
        return

    raise ValueError(f"Unsupported command: {args.command}")
