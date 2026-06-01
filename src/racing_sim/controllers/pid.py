from dataclasses import dataclass

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.model.inputs import ControlInput

def saturate(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))

@dataclass
class PIDSpeed():
    target_vx: float
    kp: float
    ki: float
    kd: float
    
    integral: float = 0.0
    previous_error: float | None = None
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        
    def update(self, state: VehicleState, params: VehicleParams, dt: float) -> ControlInput:
        
        error = self.target_vx - state.vx
        self.integral += error * dt
        
        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / dt
            
        self.previous_error = error
        
        T_set = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        T_set = saturate(T_set, params.T_min, params.T_max)
        
        T_dot_cmd = (T_set - state.T) / dt
        T_dot_cmd = saturate(T_dot_cmd, params.T_dot_min, params.T_dot_max)
        
        return ControlInput(
            delta_dot_cmd=0.0,
            T_dot_cmd=T_dot_cmd
        )
        
@dataclass
class PIDHeading:
    kn: float # weight of n error
    kmu: float # weight of mu error
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    previous_error: float | None = None
    
    def update(self, state: VehicleState, params: VehicleParams, dt: float) -> ControlInput:
        
        error = - (self.kn * state.n + self.kmu * state.mu)
        self.integral += error * dt
        
        if self.previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / dt
            
        self.previous_error = error
        
        delta_set = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        delta_set = saturate(delta_set, params.delta_min, params.delta_max)
        
        delta_dot_cmd = (delta_set - state.delta) / dt
        delta_dot_cmd = saturate(delta_dot_cmd, params.delta_dot_min, params.delta_dot_max)
        
        return ControlInput(
            delta_dot_cmd=delta_dot_cmd,
            T_dot_cmd=0.0
        )