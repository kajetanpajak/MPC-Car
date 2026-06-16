from racing_sim.model.inputs import ControlInput
from racing_sim.model.state import VehicleState
from racing_sim.model.params import VehicleParams
from racing_sim.controllers.mpc.config import MPCConfig
from acados_template import AcadosOcpSolver
import numpy as np

class AcadosMPC:
    def __init__(self, solver: AcadosOcpSolver, config: MPCConfig):
        self.solver = solver
        self.config = config
        
    def update(self, state: VehicleState, params: VehicleParams, dt, track) -> ControlInput:
        x0 = state.as_vector()
        
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        
        for i in range(self.config.N):
            s_ref = state.s + i * self.config.dt * self.config.target_vx
            kappa_ref = track.curvature(s_ref)
            
            yref = self._stage_reference(s_ref)
            self.solver.set(i, "yref", yref)
            self.solver.set(i, "p", np.array([kappa_ref]))
        
        s_terminal = state.s + self.config.N * self.config.dt * self.config.target_vx
        self.solver.set(self.config.N, 'yref', self._terminal_reference(s_terminal))
        self.solver.set(self.config.N, "p", np.array([track.curvature(s_terminal)]))
        
        status = self.solver.solve()
        if status != 0:
            return ControlInput.zero_input()
        
        u0 = self.solver.get(0, "u")
        return ControlInput(
            delta_dot_cmd=float(u0[0]),
            T_dot_cmd=float(u0[1])
        )
            
            
    def _stage_reference(self, s_ref):
        return np.array([
            s_ref,
            0.0,
            0.0,
            self.config.target_vx,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])
        
    def _terminal_reference(self, s_ref):
        return np.array([
            s_ref,
            0.0,
            0.0,
            self.config.target_vx,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
