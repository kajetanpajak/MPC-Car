from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from racing_sim.controllers.mpc.acados_model import create_vehicle_model
from racing_sim.controllers.mpc.max_progress_config import ProgressMPCConfig
from racing_sim.model.params import VehicleParams

def create_mpc_ocp(params: VehicleParams, config: ProgressMPCConfig):
    ocp = AcadosOcp()
    model = create_vehicle_model(params=params)
    ocp.model = model
    
    x = model.x
    u = model.u
    
    s = x[0]
    n = x[1]
    mu = x[2]
    vx = x[3]
    vy = x[4]
    r = x[5]
    delta = x[6]
    T = x[7]

    delta_dot_cmd = u[0]
    T_dot_cmd = u[1]

    N = config.N
    ocp.dims.N = N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * config.dt
    
    # ---------- EXTERNAL COST - MAXIMIZE s ----------
    
    step_cost = (
        - config.q_progress * s
        + config.q_n * n ** 2
        + config.q_mu * mu ** 2
        + config.q_vx * vx ** 2
        + config.q_vy * vy ** 2
        + config.q_r * r ** 2
        + config.q_delta * delta ** 2
        + config.q_T * T ** 2
        + config.r_delta_dot * delta_dot_cmd ** 2
        + config.r_T_dot * T_dot_cmd ** 2
    )
    
    terminal_cost = (
        -config.q_progress_e * s
        + config.q_n_e * n ** 2
        + config.q_mu_e * mu ** 2
    )
    
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    
    ocp.model.cost_expr_ext_cost = step_cost
    ocp.model.cost_expr_ext_cost_e = terminal_cost
    
    # ---------- CONSTRAINS ----------
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([
        params.delta_dot_min,
        params.T_dot_min
    ])
    ocp.constraints.ubu = np.array([
        params.delta_dot_max,
        params.T_dot_max
    ])
    
    
    ocp.constraints.idxbx = np.array([1, 3, 6, 7]) # n, vx, delta, T
    ocp.constraints.lbx = np.array([
        -config.track_width_right,
        config.vx_min,
        params.delta_min,
        params.T_min
    ])
    ocp.constraints.ubx = np.array([
        config.track_width_left,
        config.vx_max,
        params.delta_max,
        params.T_max
    ])
    
    # ---------- SLACK ----------
    ocp.constraints.idxsbx = np.array([0]) # slack on n
    ocp.cost.zl = np.array([config.slack_z])
    ocp.cost.Zl = np.array([config.slack_Z])
    ocp.cost.zu = np.array([config.slack_z])
    ocp.cost.Zu = np.array([config.slack_Z])

    ocp.constraints.x0 = np.zeros(model.x.rows())
    ocp.parameter_values = np.array([0.0])
    
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    
    ocp.code_export_dir = "build/acados_max_progress_mpc"
    
    return AcadosOcpSolver(
        ocp,
        json_file="build/acados_max_progress_mpc/racing_vehicle_ocp.json"
    )
    
