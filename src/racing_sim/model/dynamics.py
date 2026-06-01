import numpy as np

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.model.inputs import ControlInput
from racing_sim.model.tire_model import compute_tire_forces

def compute_longitudinal_force(params: VehicleParams, state: VehicleState) -> float:
    motor_force = params.Cm * state.T
    drag_force = params.Cr2 * state.vx ** 2
    
    longitudinal_force = motor_force - params.Cr0 - drag_force
    
    return longitudinal_force

def compute_torque_vectoring_moment(params: VehicleParams, state: VehicleState) -> float:
    r_t = np.tan(state.delta) * state.vx / (params.lF + params.lR)
    M_tv = params.ptv * (r_t - state.r)
    
    return M_tv

def vehicle_dynamics(params: VehicleParams,
                     state: VehicleState,
                     inputs: ControlInput,
                     track) -> VehicleState: # later change kappa to trajectory object
    
    tire_forces = compute_tire_forces(params, state)
    F_x = compute_longitudinal_force(params, state)
    M_tv = compute_torque_vectoring_moment(params, state)
    
    s_dot_denom = (1 - state.n * track.curvature(state.s))
    s_dot = (state.vx * np.cos(state.mu) - state.vy * np.sin(state.mu)) / s_dot_denom
    
    n_dot = state.vx * np.sin(state.mu) + state.vy * np.cos(state.mu)
    
    mu_dot = state.r - track.curvature(state.s) * s_dot
    
    vx_dot = 1 / params.m * (F_x - tire_forces.Fy_F * np.sin(state.delta) + params.m * state.vy * state.r)
    
    vy_dot = 1 / params.m * (tire_forces.Fy_R + tire_forces.Fy_F * np.cos(state.delta) - params.m * state.vx * state.r)
    
    r_dot = 1 / params.Iz * (tire_forces.Fy_F * params.lF * np.cos(state.delta) - tire_forces.Fy_R * params.lR + M_tv)
    
    delta_dot = inputs.delta_dot_cmd
    
    T_dot = inputs.T_dot_cmd
    
    return VehicleState(
        s=s_dot,
        n=n_dot,
        mu=mu_dot,
        vx=vx_dot,
        vy=vy_dot,
        r=r_dot,
        delta=delta_dot,
        T=T_dot
    )