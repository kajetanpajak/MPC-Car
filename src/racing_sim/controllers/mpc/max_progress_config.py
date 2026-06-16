from dataclasses import dataclass


@dataclass
class ProgressMPCConfig:
    dt: float = 0.05
    N: int = 30

    vx_min: float = 0.5
    vx_max: float = 15.0

    track_width_left: float = 1.0
    track_width_right: float = 1.0

    q_progress: float = 1.0

    q_n: float = 50.0
    q_mu: float = 10.0
    q_vx: float = 0.0
    q_vy: float = 2.0
    q_r: float = 2.0
    q_delta: float = 1.0
    q_T: float = 1.0

    r_delta_dot: float = 200.0
    r_T_dot: float = 100.0

    q_progress_e: float = 10.0
    q_n_e: float = 100.0
    q_mu_e: float = 20.0

    slack_z: float = 1000.0
    slack_Z: float = 100000.0