from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from racing_sim.controllers.mpc.config import MPCConfig
from racing_sim.controllers.mpc.controller import AcadosMPC
from racing_sim.controllers.mpc.ocp import create_mpc_solver
from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.sim.integrators import rk4_step
from racing_sim.track.tracks import CircleTrack, WaypointTrack, EllipseTrack
# from racing_sim.track.ellipse import EllipseTrack
from racing_sim.track.geometry import curvilinear_to_global


def make_initial_state() -> VehicleState:
    return VehicleState(
        s=0.0,
        n=0.2,
        mu=0.0,
        vx=3.0,
        vy=0.0,
        r=0.0,
        delta=0.0,
        T=0.0,
    )


def main():
    Path("build/acados_mpc").mkdir(parents=True, exist_ok=True)

    params = VehicleParams.default()
    params.delta_dot_min = -0.5
    params.delta_dot_max = 0.5
    # params.ptv = 0.0

    pts = np.array([
        [36.0, 0.0],
        [32.0, 10.0],
        [24.0, 16.0],
        [30.0, 26.0],
        [18.0, 34.0],
        [4.0, 30.0],
        [-6.0, 22.0],
        [-18.0, 30.0],
        [-32.0, 24.0],
        [-36.0, 10.0],
        [-28.0, 0.0],
        [-36.0, -12.0],
        [-28.0, -26.0],
        [-12.0, -24.0],
        [-2.0, -34.0],
        [14.0, -30.0],
        [20.0, -18.0],
        [32.0, -14.0],
        [26.0, -4.0],
        [36.0, 0.0],
    ])
    
    track = WaypointTrack(pts, 1.0, 1.0, True)
    
    # track = EllipseTrack(50, 20, 1.0, 1.0, 1000)

    config = MPCConfig(
        dt=0.05,
        N=30,
        target_vx=12.0,
        vx_max=25.0,
        track_width_left=track.width_left,
        track_width_right=track.width_right,
    )
    
    # mpc_params = VehicleParams.default()
    # mpc_params.delta_dot_min = -1.5
    # mpc_params.delta_dot_max = 1.5
    # mpc_params.DF = 0.8
    # mpc_params.DR = 0.8
    # mpc_params.CF = 1.5
    # mpc_params.CR = 1.5
    # mpc_params.DF = 8.0
    # mpc_params.DR = 8.5

    solver = create_mpc_solver(params, config)
    controller = AcadosMPC(solver, config)

    state = make_initial_state()
    dt = 0.01
    num_steps = 4000

    time_history = []
    s_history = []
    n_history = []
    mu_history = []
    vx_history = []
    vy_history = []
    r_history = []
    delta_history = []
    t_history = []
    delta_dot_history = []
    t_dot_history = []
    pos_history = []
    status_history = []

    for step in range(num_steps):
        inputs = controller.update(state, params, dt, track)

        state = rk4_step(params, state, inputs, dt, track)

        time = step * dt
        pose = curvilinear_to_global(state, track)

        time_history.append(time)
        s_history.append(state.s)
        n_history.append(state.n)
        mu_history.append(state.mu)
        vx_history.append(state.vx)
        vy_history.append(state.vy)
        r_history.append(state.r)
        delta_history.append(state.delta)
        t_history.append(state.T)
        delta_dot_history.append(inputs.delta_dot_cmd)
        t_dot_history.append(inputs.T_dot_cmd)
        pos_history.append(pose)

        print(
            f"step={step:03d} "
            f"s={state.s:7.3f} "
            f"n={state.n:7.3f} "
            f"mu={state.mu:7.3f} "
            f"vx={state.vx:7.3f} "
            f"delta_dot={inputs.delta_dot_cmd:7.3f} "
            f"T_dot={inputs.T_dot_cmd:7.3f}"
        )

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

    axes[0, 0].plot(time_history, s_history)
    axes[0, 0].set_ylabel("s [m]")
    axes[0, 0].grid()

    axes[0, 1].plot(time_history, n_history)
    axes[0, 1].axhline(track.width_left, color="r", linestyle="--")
    axes[0, 1].axhline(-track.width_right, color="r", linestyle="--")
    axes[0, 1].set_ylabel("n [m]")
    axes[0, 1].grid()

    axes[1, 0].plot(time_history, mu_history)
    axes[1, 0].set_ylabel("mu [rad]")
    axes[1, 0].grid()

    axes[1, 1].plot(time_history, vx_history, label="vx")
    axes[1, 1].plot(time_history, vy_history, label="vy")
    axes[1, 1].axhline(config.target_vx, color="k", linestyle="--", label="target vx")
    axes[1, 1].set_ylabel("velocity [m/s]")
    axes[1, 1].legend()
    axes[1, 1].grid()

    axes[2, 0].plot(time_history, r_history)
    axes[2, 0].set_ylabel("r [rad/s]")
    axes[2, 0].grid()

    axes[2, 1].plot(time_history, delta_history, label="delta")
    axes[2, 1].plot(time_history, t_history, label="T")
    axes[2, 1].set_ylabel("actuator state")
    axes[2, 1].legend()
    axes[2, 1].grid()

    axes[3, 0].plot(time_history, delta_dot_history)
    axes[3, 0].set_ylabel("delta_dot_cmd")
    axes[3, 0].set_xlabel("time [s]")
    axes[3, 0].grid()

    axes[3, 1].plot(time_history, t_dot_history)
    axes[3, 1].set_ylabel("T_dot_cmd")
    axes[3, 1].set_xlabel("time [s]")
    axes[3, 1].grid()

    fig.tight_layout()

    track_s = np.linspace(0.0, max(s_history[-1], 1.0), 1000)
    track_xy = np.array([track.position(s) for s in track_s])
    vehicle_xy = np.array([(pose.x, pose.y) for pose in pos_history])

    fig_path, ax_path = plt.subplots(figsize=(8, 8))
    ax_path.plot(track_xy[:, 0], track_xy[:, 1], label="track centerline")
    ax_path.plot(vehicle_xy[:, 0], vehicle_xy[:, 1], label="MPC path")
    ax_path.scatter(vehicle_xy[0, 0], vehicle_xy[0, 1], color="green", label="start")
    ax_path.scatter(vehicle_xy[-1, 0], vehicle_xy[-1, 1], color="red", label="end")
    ax_path.axis("equal")
    ax_path.grid()
    ax_path.legend()
    ax_path.set_title("MPC vehicle path")
    fig_path.tight_layout()

    plt.show()
    
    from racing_sim.visualization.animation import RaceReplay, Frame
    
    frames = []
    
    for i in range(len(pos_history)):
        t = i * dt
        state = VehicleState(
            s=s_history[i],
            n=n_history[i],
            mu=mu_history[i],
            vx=vx_history[i],
            vy=vy_history[i],
            r=r_history[i],
            delta=delta_history[i],
            T=t_history[i]
        )
        
        pose = pos_history[i]
        
        frames.append(Frame(
            time=t,
            state=state,
            x=pose.x,
            y=pose.y,
            psi=pose.psi,
        ))
        
        replay = RaceReplay(
            frames=frames,
            track=track,
            params=params,
            width=1200,
            height=1000,
            scale=9.0
        )
        
    replay.run()

if __name__ == "__main__":
    main()
    
