import casadi as ca
from acados_template import AcadosModel
from racing_sim.model.params import VehicleParams

def create_vehicle_model(params: VehicleParams) -> AcadosModel:
    # variables -----------------------
    x = ca.SX.sym("x", 8)
    u = ca.SX.sym("u", 2)
    xdot = ca.SX.sym("xdot", 8)
    
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
    
    p = ca.SX.sym("p", 1)
    kappa = p[0] # track curvature at prediction point
    
    vx_safe = ca.fmax(ca.fabs(vx), params.min_vx)
    
    # tire forces 
    alpha_F = ca.atan((vy + params.lF * r) / vx_safe) - delta
    alpha_R = ca.atan((vy - params.lR * r) / vx_safe)
    
    g = 9.81
    
    wheelbase = params.lR + params.lF
    
    Fn_F = (params.lR / wheelbase) * params.m * g
    Fn_R = (params.lF / wheelbase) * params.m * g
    
    Fy_F = -Fn_F * params.DF * ca.sin(params.CF * ca.atan(params.BF * alpha_F))
    Fy_R = -Fn_R * params.DR * ca.sin(params.CR * ca.atan(params.BR * alpha_R))
    
    # longitudinal force 
    F_x = params.Cm * T - params.Cr0 - params.Cr2 * vx ** 2
    
    # torque vectoring moment
    r_target = ca.tan(delta) * vx / wheelbase
    M_tv = params.ptv * (r_target - r)
    
    # state equations
    s_dot = (vx * ca.cos(mu) - vy * ca.sin(mu)) / (1.0 - n * kappa)
    
    n_dot = vx * ca.sin(mu) + vy * ca.cos(mu)
    
    mu_dot = r - kappa * s_dot
    
    vx_dot = 1 / params.m * (F_x - Fy_F * ca.sin(delta) + params.m * vy * r)
    
    vy_dot = 1 / params.m * (Fy_R + Fy_F * ca.cos(delta) - params.m * vx * r)
    
    r_dot = 1 / params.Iz * (Fy_F * params.lF * ca.cos(delta) - Fy_R * params.lR + M_tv)
    
    f_expl = ca.vertcat(
        s_dot,
        n_dot,
        mu_dot,
        vx_dot,
        vy_dot,
        r_dot,
        delta_dot_cmd,
        T_dot_cmd
    )
    
    # acados model
    model = AcadosModel()
    model.name = "racing_vehicle"
    model.x = x
    model.u = u
    model.xdot = xdot
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    
    return model

if __name__ == "__main__":
    model = create_vehicle_model(VehicleParams.default())
    print(model.f_expl_expr)
