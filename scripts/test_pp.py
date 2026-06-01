from racing_sim.controllers.pid import PIDSpeed
from racing_sim.controllers.pure_pursuit import PurePursuit
from racing_sim.model.inputs import ControlInput
from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.sim.integrators import rk4_step
from racing_sim.track.circle import CircleTrack
from racing_sim.track.geometry import curvilinear_to_global
from racing_sim.track.straight import StraightTrack
from racing_sim.track.ellipse import EllipseTrack

import matplotlib.pyplot as plt
import numpy as np


def make_initial_state() -> VehicleState:
    return VehicleState(
        s=0.0,
        n=-3.0,
        mu=-np.pi/2.0,
        vx=0.0,
        vy=0.0,
        r=0.0,
        delta=0.0,
        T=0.0,
    )


def main():
    params = VehicleParams.default()
    params.ptv = 0.0
    state = make_initial_state()

    speed_controller = PIDSpeed(
        target_vx=7.0,
        kp=0.5,
        ki=0.2,
        kd=0.100,
    )

    steering_controller = PurePursuit(
        lookahead_distance=2.0,
    )

    dt = 0.001
    # track = CircleTrack(20.0, 1.0, 1.0)
    # track = StraightTrack(1.0, 1.0)
    track = EllipseTrack(30.0, 15.0, 1.0, 1.0)
    num_steps = 30000

    time_history = []
    s_history = []
    n_history = []
    mu_history = []
    vx_history = []
    vy_history = []
    r_history = []
    delta_history = []
    T_history = []
    delta_dot_cmd_history = []
    T_dot_cmd_history = []
    pos_history = []

    for step in range(num_steps):
        speed_input = speed_controller.update(state, params, dt)
        steering_input = steering_controller.update(state, params, dt, track)

        inputs = ControlInput(
            delta_dot_cmd=steering_input.delta_dot_cmd,
            T_dot_cmd=speed_input.T_dot_cmd,
        )

        state = rk4_step(params, state, inputs, dt, track)
        time = step * dt

        time_history.append(time)
        s_history.append(state.s)
        n_history.append(state.n)
        mu_history.append(state.mu)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        r_history.append(state.r)
        delta_history.append(state.delta)
        T_history.append(state.T)
        delta_dot_cmd_history.append(inputs.delta_dot_cmd)
        T_dot_cmd_history.append(inputs.T_dot_cmd)
        pos_history.append(curvilinear_to_global(state, track))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(time_history, s_history)
    axes[0, 0].set_title("Progress s")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("s [m]")

    axes[0, 1].plot(time_history, n_history)
    axes[0, 1].set_title("Lateral Deviation n")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("n [m]")

    axes[1, 0].plot(time_history, vx_history, label="vx")
    axes[1, 0].plot(time_history, vy_history, label="vy")
    axes[1, 0].set_title("Velocities")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Velocity [m/s]")
    axes[1, 0].legend()

    axes[1, 1].plot(time_history, mu_history)
    axes[1, 1].set_title("Heading Error mu")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("mu [rad]")

    axes[2, 0].plot(time_history, r_history)
    axes[2, 0].set_title("Yaw Rate r")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_ylabel("r [rad/s]")

    axes[2, 1].plot(time_history, delta_history, label="delta")
    axes[2, 1].plot(time_history, T_history, label="T")
    axes[2, 1].set_title("Actuator States")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].legend()

    plt.tight_layout()

    if s_history:
        s_min = min(0.0, min(s_history))
        s_max = max(s_history)
    else:
        s_min = 0.0
        s_max = 1.0

    track_s = np.linspace(s_min, s_max, 1000)
    track_positions = np.array([track.position(s) for s in track_s])
    global_positions = np.array([(pose.x, pose.y) for pose in pos_history])

    fig_track, ax_track = plt.subplots(figsize=(8, 8))
    ax_track.plot(
        track_positions[:, 0],
        track_positions[:, 1],
        label="track centerline",
    )

    if len(global_positions) > 0:
        ax_track.plot(
            global_positions[:, 0],
            global_positions[:, 1],
            label="vehicle path",
        )
        ax_track.scatter(
            global_positions[0, 0],
            global_positions[0, 1],
            color="green",
            s=40,
            label="start",
        )
        ax_track.scatter(
            global_positions[-1, 0],
            global_positions[-1, 1],
            color="red",
            s=40,
            label="end",
        )

    ax_track.set_title("Pure Pursuit Global Vehicle Path")
    ax_track.set_xlabel("x [m]")
    ax_track.set_ylabel("y [m]")
    ax_track.axis("equal")
    ax_track.grid()
    ax_track.legend()
    fig_track.tight_layout()
    
    fig_inputs, ax_inputs = plt.subplots(figsize=(8, 8))
    ax_inputs.plot(time_history, delta_dot_cmd_history, label="delta_dot_cmd")
    ax_inputs.plot(time_history, T_dot_cmd_history, label="T_dot_cmd")
    ax_inputs.set_title("Control Input Signals")
    ax_inputs.set_xlabel("Time [s]")
    ax_inputs.set_ylabel("Input command")
    ax_inputs.grid()
    ax_inputs.legend()
    fig_inputs.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
