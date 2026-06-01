from dataclasses import dataclass
import numpy as np
from racing_sim.model.state import VehicleState

@dataclass
class GlobalPose:
    x: float
    y: float
    psi: float
    
def curvilinear_to_global(state: VehicleState, track) -> GlobalPose:
    x_track, y_track = track.position(state.s)
    psi_track = track.heading(state.s)
    
    x = x_track - state.n * np.sin(psi_track)
    y = y_track + state.n * np.cos(psi_track)
    psi = psi_track + state.mu
    
    return GlobalPose(x=x,
                      y=y,
                      psi=psi)
    