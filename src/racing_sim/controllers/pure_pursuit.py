from dataclasses import dataclass

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.model.inputs import ControlInput
from racing_sim.track.geometry import curvilinear_to_global

import numpy as np

def saturate(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))

@dataclass 
class PurePursuit:
    lookahead_distance: float
    
    def update(self,
               state: VehicleState,
               params: VehicleParams,
               dt: float,
               track) -> ControlInput:
        if dt <= 0.0:
            return ControlInput.zero_input()
        
        car_pos = curvilinear_to_global(state, track)
        
        lookahead_s = state.s + self.lookahead_distance
        x_la, y_la = track.position(lookahead_s) # lookahead point coords
        
        # car pose is (x, y) of CoG, must be changed to (x_rear, y_rear)
        # given in global coords
        x_rear = car_pos.x - params.lR * np.cos(car_pos.psi)
        y_rear = car_pos.y - params.lR * np.sin(car_pos.psi)
        psi_rear = car_pos.psi
        
        dx = x_la - x_rear
        dy = y_la - y_rear
        
        # lookahead point coords in car's reference frame
        x_la_r = np.cos(psi_rear) * dx + np.sin(psi_rear) * dy
        y_la_r = -np.sin(psi_rear) * dx + np.cos(psi_rear) * dy
        
        L_squared = x_la_r ** 2 + y_la_r ** 2
        
        if L_squared < 1e-9:
            delta_set = 0.0
        else:
            kappa = 2.0 * y_la_r / L_squared
            delta_set = np.arctan((params.lF + params.lR) * kappa)
            
        delta_set = saturate(delta_set, params.delta_min, params.delta_max)
            
        delta_dot_cmd = (delta_set - state.delta) / dt
        delta_dot_cmd = saturate(delta_dot_cmd, params.delta_dot_min, params.delta_dot_max)
        
        return ControlInput(
            delta_dot_cmd=delta_dot_cmd,
            T_dot_cmd=0.0
        )
