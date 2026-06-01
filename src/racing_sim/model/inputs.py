from dataclasses import dataclass

import numpy as np 

@dataclass
class ControlInput:
    delta_dot_cmd: float
    T_dot_cmd: float
    
    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.delta_dot_cmd,
            self.T_dot_cmd],
            dtype=float
        )
        
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "ControlInput":
        values = np.asarray(vector, dtype=float).reshape(2)
        return cls(
            delta_dot_cmd=values[0],
            T_dot_cmd=values[1]
        )
        
    @classmethod
    def zero_input(cls) -> "ControlInput":
        return cls(
            delta_dot_cmd=0.0,
            T_dot_cmd=0.0
        )