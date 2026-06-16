from dataclasses import dataclass

@dataclass
class MPCConfig:
    dt: float = 0.05
    N: int = 30
    
    target_vx: float = 7.0
    vx_max: float = 15.0
    
    track_width_left: float = 1.0
    track_width_right: float = 1.0
    
    # state weights
    #           s,    n,    mu,   vx,  vy,  r, delta, T
    Q: tuple = (0.0, 20.0, 10.0, 30.0, 1.0, 2.0, 0.0, 0.0)
    # input weights: delta_dot_cmd, T_dot_cmd
    R: tuple = (10000.0, 5000.0)
    # terminal weights
    #             s,    n,    mu,   vx,  vy,  r, delta, T
    Qe: tuple = (0.0, 40.0, 20.0, 50.0, 2.0, 5.0, 0.0, 0.0)