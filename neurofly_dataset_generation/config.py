"""Project-wide defaults for simulation, outputs, and validation."""

RADIUS_START = 2.0
RADIUS_RATE = 0.15
OMEGA = 1.0
V_Z = 0.5
RAMP_TIME = 8.0
SIM_RATE = 200
T_FINAL = 20.0

POSITION_RMSE_LIMIT = 0.10
POSITION_MAX_LIMIT = 0.25
MOTOR_SPEED_MAX_LIMIT = 2400.0

DEFAULT_SIM_CSV = "basic_usage.csv"
DEFAULT_SENSOR_CSV = "spiral_flight_data.csv"
DEFAULT_NOISY_CSV = "basic_usage_noisy.csv"
DEFAULT_PLOT_DIR = "trajectory_plots"
