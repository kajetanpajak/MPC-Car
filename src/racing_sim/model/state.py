from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleState:
    # Curvilinear coordinates
    s: float
    n: float
    mu: float

    # Body-frame velocities and yaw rate
    vx: float
    vy: float
    r: float

    # Actuator states
    delta: float
    T: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.s,
                self.n,
                self.mu,
                self.vx,
                self.vy,
                self.r,
                self.delta,
                self.T,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "VehicleState":
        values = np.asarray(vector, dtype=float).reshape(8)
        return cls(
            s=values[0],
            n=values[1],
            mu=values[2],
            vx=values[3],
            vy=values[4],
            r=values[5],
            delta=values[6],
            T=values[7],
        )
