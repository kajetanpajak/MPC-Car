from dataclasses import dataclass

import numpy as np
from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState

@dataclass
class TireForces:
    """
    Dataclass with tire forces and slip angles.
    """
    # slip angles
    alpha_F: float
    alpha_R: float
    
    # lateral forces 
    Fy_F: float
    Fy_R: float
    
    # normal forces
    Fn_F: float
    Fn_R: float
    
def compute_normal_forces(params: VehicleParams) -> tuple[float, float]:
    """
    Computes the tires' normal forces
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :return: Normal forces: [Front, Rear].
    :rtype: tuple[float, float]
    """
    g = 9.81
    
    wheel_distance = params.lF + params.lR
    
    Fn_F = (params.lR / wheel_distance) * params.m * g
    Fn_R = (params.lF / wheel_distance) * params.m * g 
    return Fn_F, Fn_R

def compute_slip_angles(params: VehicleParams, state: VehicleState) ->tuple[float, float]:
    """
    Computes the tires' slip angles.
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :param state: Car's state.
    :type state: VehicleState
    :return: Slip angles: [Front, Rear].
    :rtype: tuple[float, float]
    """
    vx_safe = max(abs(state.vx), params.min_vx)
    
    alpha_F = np.arctan((state.vy + params.lF * state.r) / vx_safe) - state.delta
    alpha_R = np.arctan((state.vy - params.lR * state.r) / vx_safe)
    
    return alpha_F, alpha_R

def compute_lateral_forces(alpha_F: float,
                           alpha_R: float,
                           Fn_F: float,
                           Fn_R: float,
                           params: VehicleParams) -> tuple[float, float]:
    """
    Computes the tires' lateral forces.
    
    :param alpha_F: Front tire slip angle.
    :type alpha_F: float
    :param alpha_R: Rear tire slip angle.
    :type alpha_R: float
    :param Fn_F: Front tire normal force.
    :type Fn_F: float
    :param Fn_R: Rear tire normal force.
    :type Fn_R: float
    :param params: Car's paramters
    :type params: VehicleParams
    :return: Lateral forces: [Front, Rear].
    :rtype: tuple[float, float]
    """
    Fy_F = -Fn_F * params.DF * np.sin(params.CF * np.arctan(params.BF * alpha_F))
    Fy_R = -Fn_R * params.DR * np.sin(params.CR * np.arctan(params.BR * alpha_R))
    
    return Fy_F, Fy_R

def compute_tire_forces(params: VehicleParams, state: VehicleState) -> TireForces:
    """
    Wrapper to compute all tire forces and slip angles.
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :param state: Car's state.
    :type state: VehicleState
    :return: Dataclass with tire forces and slip angles
    :rtype: TireForces
    """
    Fn_F, Fn_R = compute_normal_forces(params)
    alpha_F, alpha_R = compute_slip_angles(params, state)
    Fy_F, Fy_R = compute_lateral_forces(alpha_F, alpha_R, Fn_F, Fn_R, params)
    
    return TireForces(
        alpha_F=alpha_F,
        alpha_R=alpha_R,
        Fy_F=Fy_F,
        Fy_R=Fy_R,
        Fn_F=Fn_F,
        Fn_R=Fn_R
    )
    
    
