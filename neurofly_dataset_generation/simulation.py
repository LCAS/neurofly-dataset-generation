"""Simulation orchestration for the dataset-generation workflow."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.environments import Environment
from rotorpy.simulate import ExitStatus
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor

from .config import (
    DEFAULT_SENSOR_CSV,
    DEFAULT_SIM_CSV,
    FIGURE_EIGHT_OMEGA,
    FIGURE_EIGHT_X_SCALE,
    FIGURE_EIGHT_Y_SCALE,
    FIGURE_EIGHT_Z_AMPLITUDE,
    FIGURE_EIGHT_Z_BASE,
    MOTOR_SPEED_MAX_LIMIT,
    OMEGA,
    POSITION_MAX_LIMIT,
    POSITION_RMSE_LIMIT,
    RADIUS_RATE,
    RADIUS_START,
    RAMP_TIME,
    SIM_RATE,
    T_FINAL,
    TRAJECTORY_CHOICES,
    V_Z,
)
from .plotting import plot_trajectory_comparison
from .trajectories import (
    RampedConicalSpiralTrajectory,
    RampedFigureEightVariableAltitudeTrajectory,
)

def hover_rotor_speed(params):
    return np.sqrt((params["mass"] * 9.81 / params["num_rotors"]) / params["k_eta"])


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_only_quaternion(yaw):
    return Rotation.from_euler("z", yaw).as_quat()


def quaternion_yaw(quaternion):
    return Rotation.from_quat(quaternion).as_euler("zyx")[0]


def build_trajectory(trajectory_name="conical-spiral"):
    if trajectory_name == "figure-eight":
        return RampedFigureEightVariableAltitudeTrajectory(
            x_scale=FIGURE_EIGHT_X_SCALE,
            y_scale=FIGURE_EIGHT_Y_SCALE,
            z_base=FIGURE_EIGHT_Z_BASE,
            z_amplitude=FIGURE_EIGHT_Z_AMPLITUDE,
            omega=FIGURE_EIGHT_OMEGA,
            ramp_time=RAMP_TIME,
        )

    if trajectory_name != "conical-spiral":
        raise ValueError(
            f"Unsupported trajectory '{trajectory_name}'. "
            f"Choose one of: {', '.join(TRAJECTORY_CHOICES)}."
        )

    return RampedConicalSpiralTrajectory(
        radius_start=RADIUS_START,
        radius_rate=RADIUS_RATE,
        omega=OMEGA,
        v_z=V_Z,
        ramp_time=RAMP_TIME,
    )


def build_initial_state(trajectory):
    start_state = trajectory.update(0.0)
    hover_speed = hover_rotor_speed(quad_params)

    return {
        "x": start_state["x"],
        "v": np.zeros(3),
        "q": yaw_only_quaternion(start_state["yaw"]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.full(quad_params["num_rotors"], hover_speed),
    }


def export_sensor_csv(results, savepath):
    csv_data = np.column_stack(
        (
            results["time"],
            results["state"]["x"],
            results["imu_measurements"]["accel"],
            results["imu_measurements"]["gyro"],
        )
    )
    headers = "time,pos_x,pos_y,pos_z,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"
    np.savetxt(savepath, csv_data, delimiter=",", header=headers, comments="")


def validate_reference_consistency(trajectory, dt=1e-4):
    sample_times = [0.5 * RAMP_TIME, RAMP_TIME + 2.0]

    for sample_time in sample_times:
        state_prev = trajectory.update(sample_time - dt)
        state_now = trajectory.update(sample_time)
        state_next = trajectory.update(sample_time + dt)

        velocity_fd = (state_next["x"] - state_prev["x"]) / (2.0 * dt)
        accel_fd = (state_next["x_dot"] - state_prev["x_dot"]) / (2.0 * dt)

        if not np.allclose(state_now["x_dot"], velocity_fd, atol=5e-3, rtol=1e-2):
            raise RuntimeError(
                f"x_dot finite-difference check failed at t={sample_time:.2f}s: "
                f"analytic={state_now['x_dot']}, fd={velocity_fd}"
            )

        if not np.allclose(state_now["x_ddot"], accel_fd, atol=5e-2, rtol=1e-2):
            raise RuntimeError(
                f"x_ddot finite-difference check failed at t={sample_time:.2f}s: "
                f"analytic={state_now['x_ddot']}, fd={accel_fd}"
            )


def validate_results(results):
    if results["exit"] != ExitStatus.TIMEOUT:
        raise RuntimeError(
            f"Simulation exited with '{results['exit'].value}' instead of timeout."
        )

    position_error = np.linalg.norm(results["state"]["x"] - results["flat"]["x"], axis=1)
    position_rmse = np.sqrt(np.mean(position_error**2))
    position_max = np.max(position_error)
    motor_speed_max = np.max(results["state"]["rotor_speeds"])

    if position_rmse > POSITION_RMSE_LIMIT:
        raise RuntimeError(
            f"Position RMSE {position_rmse:.4f} m exceeds {POSITION_RMSE_LIMIT:.2f} m."
        )

    if position_max > POSITION_MAX_LIMIT:
        raise RuntimeError(
            f"Maximum position error {position_max:.4f} m exceeds {POSITION_MAX_LIMIT:.2f} m."
        )

    if motor_speed_max > MOTOR_SPEED_MAX_LIMIT:
        raise RuntimeError(
            f"Maximum motor speed {motor_speed_max:.2f} rad/s exceeds "
            f"{MOTOR_SPEED_MAX_LIMIT:.1f} rad/s."
        )

    initial_speed = np.linalg.norm(results["state"]["v"][0])
    initial_yaw = quaternion_yaw(results["state"]["q"][0])
    reference_yaw = results["flat"]["yaw"][0]
    initial_yaw_error = abs(wrap_angle(initial_yaw - reference_yaw))

    if initial_speed > 1e-9:
        raise RuntimeError(
            f"Initial translational speed is not near zero: {initial_speed:.6e} m/s."
        )

    if initial_yaw_error > 1e-6:
        raise RuntimeError(
            f"Initial yaw does not match the tangent heading: {initial_yaw_error:.6e} rad."
        )

    print(
        "Validation metrics:",
        {
            "exit": results["exit"].value,
            "position_rmse_m": float(position_rmse),
            "position_max_m": float(position_max),
            "motor_speed_max_rad_s": float(motor_speed_max),
            "initial_speed_m_s": float(initial_speed),
            "initial_yaw_error_rad": float(initial_yaw_error),
        },
    )


def run_spiral_study(
    plot=True,
    animate=True,
    verbose=True,
    plot_tracking=False,
    tracking_output_dir=None,
    sensor_csv_path=DEFAULT_SENSOR_CSV,
    simulation_csv_path=DEFAULT_SIM_CSV,
    trajectory_name="conical-spiral",
):
    trajectory = build_trajectory(trajectory_name)
    validate_reference_consistency(trajectory)

    sim_instance = Environment(
        vehicle=Multirotor(quad_params),
        controller=SE3Control(quad_params),
        trajectory=trajectory,
        sim_rate=SIM_RATE,
    )
    sim_instance.vehicle.initial_state = build_initial_state(trajectory)

    results = sim_instance.run(
        t_final=T_FINAL,
        use_mocap=False,
        terminate=False,
        plot=plot,
        plot_mocap=True,
        plot_estimator=True,
        plot_imu=True,
        animate_bool=animate,
        animate_wind=True,
        verbose=verbose,
        fname=None,
    )

    validate_results(results)
    export_sensor_csv(results, sensor_csv_path)
    sim_instance.save_to_csv(simulation_csv_path)

    if plot_tracking:
        plot_trajectory_comparison(
            results=results,
            show=not plot and not animate,
            output_dir=tracking_output_dir,
        )

    print(
        f"Simulation complete! {trajectory_name} trajectory executed and CSV data exported."
    )
    return results
