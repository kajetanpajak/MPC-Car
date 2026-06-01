from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.model.inputs import ControlInput
from racing_sim.sim.integrators import rk4_step
from racing_sim.track.circle import CircleTrack
from racing_sim.track.straight import StraightTrack

import matplotlib.pyplot as plt
import numpy as np

def make_initial_state() -> VehicleState:
    return VehicleState(
        s=0.0,
        n=0.0,
        mu=0.0,
        vx=2.0,
        vy=0.0,
        r=0.0,
        delta=0.15,
        T=0.043
    )

def main():
    params = VehicleParams.default()
    params.ptv = 0.0
    state = make_initial_state()
    
    inputs = ControlInput(
        delta_dot_cmd=0.0,
        T_dot_cmd=0.0
    )
    
    dt = 0.01
    track = CircleTrack(10.0, 1.0, 1.0)
    # track = StraightTrack(1.0, 1.0)
    num_steps = 5_000
    
    time_history = []
    s_history = []
    n_history = []
    mu_history = []
    vx_history = []
    vy_history = []
    r_history = []
    delta_history = []
    T_history = []

    
    for step in range(num_steps):
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
        
        # r_t = np.tan(state.delta) * state.vx / (params.lF + params.lR)
        # print(f"Tire forces: {}")
        # print(r_t)

        
        # if step % 50 == 0:
        #     print(
        #         f"step={step:4d} "
        #         f"s={state.s:7.3f} "
        #         f"n={state.n:7.3f} "
        #         f"mu={state.mu:7.3f} "
        #         f"vx={state.vx:7.3f} "
        #         f"vy={state.vy:7.3f} "
        #         f"r={state.r:7.3f} "
        #         f"delta={state.delta:7.3f} "
        #         f"T={state.T:7.3f}"
        #     )
            
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
    plt.show()

    
if __name__ == "__main__":
    main()