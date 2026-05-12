"""Noise injection and derived state estimation utilities."""

import numpy as np
from scipy.spatial.transform import Rotation

from .io_utils import (
    default_output_path,
    read_csv,
    require_columns,
    rows_to_numeric_matrix,
    select_columns,
    write_csv,
)


ACCEL_SIGMA_WHITE = 0.12
ACCEL_SIGMA_RW = 0.002
GYRO_SIGMA_WHITE = 0.008
GYRO_SIGMA_RW = 0.00015

MOCAP_POSITION_SIGMAS = {
    "mocap_x": 0.01,
    "mocap_y": 0.01,
    "mocap_z": 0.015,
}
MOCAP_VELOCITY_SIGMAS = {
    "mocap_xdot": 0.03,
    "mocap_ydot": 0.03,
    "mocap_zdot": 0.04,
}
MOCAP_ATTITUDE_SIGMA_AXIS_RAD = np.deg2rad(0.3)

GPS_POSITION_SIGMAS = {
    "gps_x": 0.35,
    "gps_y": 0.35,
    "gps_z": 0.5,
}
GPS_BIAS_SIGMA_RW = {
    "gps_x": 0.01,
    "gps_y": 0.01,
    "gps_z": 0.015,
}
GPS_BIAS_RHO = 0.995

WIND_SIGMAS = {
    "windx": 0.05,
    "windy": 0.05,
    "windz": 0.05,
}
ROTOR_SPEED_SIGMA_REL = 0.007
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])

TIME_COLUMN = "time"
POSITION_COLUMNS = ("x", "y", "z")
VELOCITY_COLUMNS = ("xdot", "ydot", "zdot")
STATE_QUATERNION_COLUMNS = ("qx", "qy", "qz", "qw")
ACCEL_COLUMNS = ("ax", "ay", "az")
GYRO_COLUMNS = ("gx", "gy", "gz")
ACCEL_TRUTH_COLUMNS = ("ax_gt", "ay_gt", "az_gt")
GYRO_TRUTH_COLUMNS = ("wx", "wy", "wz")
MOCAP_QUATERNION_COLUMNS = ("mocap_qx", "mocap_qy", "mocap_qz", "mocap_qw")
ROTOR_SPEED_COLUMNS = ("r1", "r2", "r3", "r4")


def random_walk_bias(rng, count, sigma_rw):
    if sigma_rw == 0.0:
        return np.zeros(count)
    steps = rng.normal(loc=0.0, scale=sigma_rw, size=count)
    return np.cumsum(steps)


def gauss_markov_bias(rng, count, sigma_rw, rho):
    bias = np.zeros(count)
    if count <= 1 or sigma_rw == 0.0:
        return bias

    for index in range(1, count):
        bias[index] = rho * bias[index - 1] + rng.normal(0.0, sigma_rw)
    return bias


def append_white_plus_rw_noise(
    rng, matrix, header_index, source_columns, output_columns, sigma_white, sigma_rw
):
    appended = {}
    for source_column, output_column in zip(source_columns, output_columns):
        values = matrix[:, header_index[source_column]]
        bias = random_walk_bias(rng, len(values), sigma_rw)
        white = rng.normal(loc=0.0, scale=sigma_white, size=len(values))
        appended[f"{output_column}_bias_rw"] = bias
        appended[f"{output_column}_noisy"] = values + bias + white
    return appended


def append_white_noise(rng, matrix, header_index, sigma_by_column):
    appended = {}
    for column, sigma in sigma_by_column.items():
        values = matrix[:, header_index[column]]
        noise = rng.normal(loc=0.0, scale=sigma, size=len(values))
        appended[f"{column}_noisy"] = values + noise
    return appended


def append_multiplicative_noise(rng, matrix, header_index, columns, sigma_rel):
    appended = {}
    for column in columns:
        values = matrix[:, header_index[column]]
        scale = 1.0 + rng.normal(loc=0.0, scale=sigma_rel, size=len(values))
        appended[f"{column}_noisy"] = values * scale
    return appended


def append_quaternion_noise(rng, matrix, header_index, sigma_axis_rad):
    quaternions = select_columns(matrix, header_index, MOCAP_QUATERNION_COLUMNS)
    rotation_vectors = rng.normal(
        loc=0.0,
        scale=sigma_axis_rad,
        size=(len(quaternions), 3),
    )
    base_rotations = Rotation.from_quat(quaternions)
    perturbations = Rotation.from_rotvec(rotation_vectors)
    noisy_quaternions = (perturbations * base_rotations).as_quat()

    return {
        "mocap_qx_noisy": noisy_quaternions[:, 0],
        "mocap_qy_noisy": noisy_quaternions[:, 1],
        "mocap_qz_noisy": noisy_quaternions[:, 2],
        "mocap_qw_noisy": noisy_quaternions[:, 3],
        "mocap_q_noise_rx_rad": rotation_vectors[:, 0],
        "mocap_q_noise_ry_rad": rotation_vectors[:, 1],
        "mocap_q_noise_rz_rad": rotation_vectors[:, 2],
    }


def append_gps_position_noise(rng, matrix, header_index, noise_scale):
    appended = {}
    position_truth = select_columns(matrix, header_index, POSITION_COLUMNS)

    for axis_index, axis_name in enumerate(("gps_x", "gps_y", "gps_z")):
        sigma_white = GPS_POSITION_SIGMAS[axis_name] * noise_scale
        sigma_rw = GPS_BIAS_SIGMA_RW[axis_name] * noise_scale
        bias = gauss_markov_bias(rng, len(position_truth), sigma_rw, GPS_BIAS_RHO)
        white = rng.normal(loc=0.0, scale=sigma_white, size=len(position_truth))
        appended[f"{axis_name}_bias"] = bias
        appended[f"{axis_name}_noisy"] = position_truth[:, axis_index] + bias + white

    return appended


def append_dead_reckoned_state(matrix, header_index, noisy_columns):
    time = matrix[:, header_index[TIME_COLUMN]]
    position_truth = select_columns(matrix, header_index, POSITION_COLUMNS)
    velocity_truth = select_columns(matrix, header_index, VELOCITY_COLUMNS)
    quaternion_truth = select_columns(matrix, header_index, STATE_QUATERNION_COLUMNS)

    accel_noisy = np.column_stack([noisy_columns[f"{column}_noisy"] for column in ACCEL_COLUMNS])
    gyro_noisy = np.column_stack([noisy_columns[f"{column}_noisy"] for column in GYRO_COLUMNS])

    sample_count = len(time)
    est_position = np.zeros((sample_count, 3))
    est_velocity = np.zeros((sample_count, 3))
    est_quaternion = np.zeros((sample_count, 4))

    est_position[0] = position_truth[0]
    est_velocity[0] = velocity_truth[0]
    est_quaternion[0] = quaternion_truth[0]
    current_rotation = Rotation.from_quat(est_quaternion[0])

    for index in range(sample_count - 1):
        dt = time[index + 1] - time[index]
        if dt < 0.0:
            raise ValueError("Time column must be monotonically non-decreasing.")

        delta_rotation = Rotation.from_rotvec(gyro_noisy[index] * dt)
        next_rotation = current_rotation * delta_rotation
        next_quaternion = next_rotation.as_quat()
        next_quaternion /= np.linalg.norm(next_quaternion)
        next_rotation = Rotation.from_quat(next_quaternion)

        world_accel = next_rotation.apply(accel_noisy[index]) + GRAVITY_WORLD
        next_velocity = est_velocity[index] + world_accel * dt
        next_position = (
            est_position[index]
            + est_velocity[index] * dt
            + 0.5 * world_accel * dt * dt
        )

        est_velocity[index + 1] = next_velocity
        est_position[index + 1] = next_position
        est_quaternion[index + 1] = next_quaternion
        current_rotation = next_rotation

    return {
        "est_x_drift": est_position[:, 0],
        "est_y_drift": est_position[:, 1],
        "est_z_drift": est_position[:, 2],
        "est_xdot_drift": est_velocity[:, 0],
        "est_ydot_drift": est_velocity[:, 1],
        "est_zdot_drift": est_velocity[:, 2],
        "est_qx_drift": est_quaternion[:, 0],
        "est_qy_drift": est_quaternion[:, 1],
        "est_qz_drift": est_quaternion[:, 2],
        "est_qw_drift": est_quaternion[:, 3],
    }


def generate_noisy_csv(
    input_path,
    output_path=None,
    mode="deadreckon",
    abs_sensor="mocap",
    seed=7,
    noise_scale=1.0,
):
    output_path = output_path or default_output_path(input_path)

    header, rows = read_csv(input_path)
    numeric_matrix = rows_to_numeric_matrix(rows)
    header_index = {name: idx for idx, name in enumerate(header)}

    require_columns(
        header_index,
        (
            TIME_COLUMN,
            *POSITION_COLUMNS,
            *VELOCITY_COLUMNS,
            *STATE_QUATERNION_COLUMNS,
            *ACCEL_TRUTH_COLUMNS,
            *GYRO_TRUTH_COLUMNS,
            *WIND_SIGMAS.keys(),
            *ROTOR_SPEED_COLUMNS,
        ),
    )

    if abs_sensor == "mocap":
        require_columns(
            header_index,
            (
                *MOCAP_POSITION_SIGMAS.keys(),
                *MOCAP_VELOCITY_SIGMAS.keys(),
                *MOCAP_QUATERNION_COLUMNS,
            ),
        )

    rng = np.random.default_rng(seed)
    noisy_columns = {}
    noisy_columns.update(
        append_white_plus_rw_noise(
            rng,
            numeric_matrix,
            header_index,
            ACCEL_TRUTH_COLUMNS,
            ACCEL_COLUMNS,
            ACCEL_SIGMA_WHITE * noise_scale,
            ACCEL_SIGMA_RW * noise_scale,
        )
    )
    noisy_columns.update(
        append_white_plus_rw_noise(
            rng,
            numeric_matrix,
            header_index,
            GYRO_TRUTH_COLUMNS,
            GYRO_COLUMNS,
            GYRO_SIGMA_WHITE * noise_scale,
            GYRO_SIGMA_RW * noise_scale,
        )
    )

    if abs_sensor == "mocap":
        noisy_columns.update(
            append_white_noise(
                rng,
                numeric_matrix,
                header_index,
                {
                    column: sigma * noise_scale
                    for column, sigma in MOCAP_POSITION_SIGMAS.items()
                },
            )
        )
        noisy_columns.update(
            append_white_noise(
                rng,
                numeric_matrix,
                header_index,
                {
                    column: sigma * noise_scale
                    for column, sigma in MOCAP_VELOCITY_SIGMAS.items()
                },
            )
        )
        noisy_columns.update(
            append_quaternion_noise(
                rng,
                numeric_matrix,
                header_index,
                MOCAP_ATTITUDE_SIGMA_AXIS_RAD * noise_scale,
            )
        )
    elif abs_sensor == "gps":
        noisy_columns.update(
            append_gps_position_noise(rng, numeric_matrix, header_index, noise_scale)
        )

    noisy_columns.update(
        append_white_noise(
            rng,
            numeric_matrix,
            header_index,
            {column: sigma * noise_scale for column, sigma in WIND_SIGMAS.items()},
        )
    )
    noisy_columns.update(
        append_multiplicative_noise(
            rng,
            numeric_matrix,
            header_index,
            ROTOR_SPEED_COLUMNS,
            ROTOR_SPEED_SIGMA_REL * noise_scale,
        )
    )

    if mode != "deadreckon":
        raise ValueError(f"Unsupported mode: {mode}")
    noisy_columns.update(
        append_dead_reckoned_state(numeric_matrix, header_index, noisy_columns)
    )

    appended_header = header + list(noisy_columns.keys())
    output_rows = []
    for row_index, row in enumerate(rows):
        appended_values = [noisy_columns[column][row_index] for column in noisy_columns]
        output_rows.append(row + [f"{value:.16g}" for value in appended_values])

    write_csv(output_path, appended_header, output_rows)
    return output_path
