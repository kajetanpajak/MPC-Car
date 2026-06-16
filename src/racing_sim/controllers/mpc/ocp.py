from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from racing_sim.controllers.mpc.acados_model import create_vehicle_model
from racing_sim.controllers.mpc.config import MPCConfig
from racing_sim.model.params import VehicleParams

def create_mpc_solver(params: VehicleParams, config: MPCConfig):
    ocp = AcadosOcp()
    model = create_vehicle_model(params)
    ocp.model = model
    
    nx = model.x.rows()
    nu = model.u.rows()
    
    N = config.N
    
    ocp.dims.N = N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * config.dt
    
    ny = nx + nu
    
    # koszt na maksymalizację s lub rzutowanie predkosci na tor
    # -s tak właściwie
    
    # linear ls cost: 0.5 * || Vx*x + Vu*u - yref ||_W^2
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    
    ocp.cost.Vx_e = np.eye(nx)
    
    ocp.cost.W = np.diag(config.Q + config.R)
    ocp.cost.W_e = np.diag(config.Qe)
    
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)
    
    # input constraints
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([params.delta_dot_min, params.T_dot_min])
    ocp.constraints.ubu = np.array([params.delta_dot_max, params.T_dot_max])
    
    
    # state constraints on n, vx, delta, T
    ocp.constraints.idxbx = np.array([1, 3, 6, 7])
    ocp.constraints.lbx = np.array([
        -config.track_width_right,
        params.min_vx,
        params.delta_min,
        params.T_min
    ])
    ocp.constraints.ubx = np.array([
        config.track_width_left,
        config.vx_max,
        params.delta_max,
        params.T_max
    ])
    
    
    # ocp.constraints.idxsbx = np.array([0])
    
    # ocp.cost.zl = np.array([1000.0])
    # ocp.cost.Zl = np.array([10000.0])

    # ocp.cost.zu = np.array([1000.0])
    # ocp.cost.Zu = np.array([10000.0])
    
    ocp.constraints.x0 = np.zeros(nx)
    
    ocp.parameter_values = np.array([0.0])
    
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    
    ocp.code_export_dir = "build/acados_mpc"
    
    return AcadosOcpSolver(
        ocp,
        json_file="build/acados_mpc/racing_vehicle_ocp.json"
    )
    
    
    
if __name__ == "__main__":
    solver = create_mpc_solver(VehicleParams.default(), MPCConfig())
