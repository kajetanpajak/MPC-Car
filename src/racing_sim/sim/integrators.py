import numpy as np

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.model.inputs import ControlInput
from racing_sim.model.dynamics import vehicle_dynamics


def rk4_step(params: VehicleParams,
             state: VehicleState,
             inputs: ControlInput,
             dt: float,
             track) -> VehicleState: 
    x = state.as_vector() # current state
    
    k1 = vehicle_dynamics(params, state, inputs, track).as_vector()
    
    k2 = vehicle_dynamics(
        params,
        VehicleState.from_vector(x + 0.5 * dt * k1),
        inputs,
        track
    ).as_vector()
    
    k3 = vehicle_dynamics(
        params,
        VehicleState.from_vector(x + 0.5 * dt * k2),
        inputs,
        track
    ).as_vector()
    
    k4 = vehicle_dynamics(
        params,
        VehicleState.from_vector(x + dt * k3),
        inputs,
        track
    ).as_vector()
    
    x_next = x + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return VehicleState.from_vector(x_next)
    