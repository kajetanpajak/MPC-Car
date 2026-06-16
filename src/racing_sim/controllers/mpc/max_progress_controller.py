from racing_sim.model.inputs import ControlInput
from racing_sim.model.state import VehicleState
from racing_sim.model.params import VehicleParams
from racing_sim.controllers.mpc.max_progress_config import ProgressMPCConfig
from acados_template import AcadosOcpSolver
import numpy as np


class MaxProgressMPC:
    def __init__(self, solver: AcadosOcpSolver, config: ProgressMPCConfig):
        self.solver = solver
        self.config = config
        self.last_status = 0
        
    def update(self, state: VehicleState, params: VehicleParams, dt, track) -> ControlInput:
        x0 = state.as_vector()
        
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        
        for i in range(self.config.N):
            s_pred = self._predict_s_for_stage(state, i)
            kappa_ref = track.curvature(s_pred)
            self.solver.set(i, "p", np.array([kappa_ref]))
        
        s_terminal = self._predict_s_for_stage(state, self.config.N)
        self.solver.set(self.config.N, "p", np.array([track.curvature(s_terminal)]))
        
        status = self.solver.solve()
        self.last_status = status
        if status != 0:
            return ControlInput.zero_input()
        
        u0 = self.solver.get(0, "u")
        return ControlInput(
            delta_dot_cmd=float(u0[0]),
            T_dot_cmd=float(u0[1])
        )

    def _predict_s_for_stage(self, state: VehicleState, stage: int) -> float:
        vx_guess = max(float(state.vx), self.config.vx_min)
        return float(state.s + stage * self.config.dt * vx_guess)