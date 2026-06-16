from dataclasses import dataclass

import numpy as np
from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState

@dataclass
class TireForces:
    """
    Dataclass with tire forces and slip angles.
    """
    # slip angles
    alpha_F: float
    alpha_R: float
    
    # lateral forces 
    Fy_F: float
    Fy_R: float
    
    # normal forces
    Fn_F: float
    Fn_R: float
    
def compute_normal_forces(params: VehicleParams) -> tuple[float, float]:
    """
    Computes the tires' normal forces
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :return: Normal forces: [Front, Rear].
    :rtype: tuple[float, float]
    """
    g = 9.81
    
    wheel_distance = params.lF + params.lR
    
    Fn_F = (params.lR / wheel_distance) * params.m * g
    Fn_R = (params.lF / wheel_distance) * params.m * g 
    return Fn_F, Fn_R

def compute_slip_angles(params: VehicleParams, state: VehicleState) ->tuple[float, float]:
    """
    Computes the tires' slip angles - the difference between where the wheel is pointing
    and where the wheel is actually moving.
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :param state: Car's state.
    :type state: VehicleState
    :return: Slip angles: [Front, Rear].
    :rtype: tuple[float, float]
    """
    vx_safe = max(abs(state.vx), params.min_vx)
    
    alpha_F = np.arctan((state.vy + params.lF * state.r) / vx_safe) - state.delta
    alpha_R = np.arctan((state.vy - params.lR * state.r) / vx_safe)
    
    return alpha_F, alpha_R

def compute_lateral_forces(alpha_F: float,
                           alpha_R: float,
                           Fn_F: float,
                           Fn_R: float,
                           params: VehicleParams) -> tuple[float, float]:
    """
    Computes the tires' lateral forces.
    
    :param alpha_F: Front tire slip angle.
    :type alpha_F: float
    :param alpha_R: Rear tire slip angle.
    :type alpha_R: float
    :param Fn_F: Front tire normal force.
    :type Fn_F: float
    :param Fn_R: Rear tire normal force.
    :type Fn_R: float
    :param params: Car's paramters
    :type params: VehicleParams
    :return: Lateral forces: [Front, Rear].
    :rtype: tuple[float, float]
    """
    Fy_F = -Fn_F * params.DF * np.sin(params.CF * np.arctan(params.BF * alpha_F))
    Fy_R = -Fn_R * params.DR * np.sin(params.CR * np.arctan(params.BR * alpha_R))
    
    return Fy_F, Fy_R

def compute_tire_forces(params: VehicleParams, state: VehicleState) -> TireForces:
    """
    Wrapper to compute all tire forces and slip angles.
    
    :param params: Car's parameters.
    :type params: VehicleParams
    :param state: Car's state.
    :type state: VehicleState
    :return: Dataclass with tire forces and slip angles
    :rtype: TireForces
    """
    Fn_F, Fn_R = compute_normal_forces(params)
    alpha_F, alpha_R = compute_slip_angles(params, state)
    Fy_F, Fy_R = compute_lateral_forces(alpha_F, alpha_R, Fn_F, Fn_R, params)
    
    return TireForces(
        alpha_F=alpha_F,
        alpha_R=alpha_R,
        Fy_F=Fy_F,
        Fy_R=Fy_R,
        Fn_F=Fn_F,
        Fn_R=Fn_R
    )

def tire_curves(params, alpha_rad):
    fn_f, fn_r = compute_normal_forces(params)

    fy_f = []
    fy_r = []

    for alpha in alpha_rad:
        front, rear = compute_lateral_forces(
            alpha_F=alpha,
            alpha_R=alpha,
            Fn_F=fn_f,
            Fn_R=fn_r,
            params=params,
        )
        fy_f.append(front)
        fy_r.append(rear)

    return np.array(fy_f), np.array(fy_r)

def main():
    params = VehicleParams.default()

    alpha_deg = np.linspace(-20.0, 20.0, 500)
    alpha_rad = np.deg2rad(alpha_deg)

    fy_f, fy_r = tire_curves(params, alpha_rad)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(bottom=0.35)

    front_line, = ax.plot(alpha_deg, fy_f, label="front")
    rear_line, = ax.plot(alpha_deg, fy_r, label="rear")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("slip angle alpha [deg]")
    ax.set_ylabel("lateral force Fy [N]")
    ax.grid()
    ax.legend()

    ax_bf = plt.axes([0.15, 0.25, 0.75, 0.03])
    ax_cf = plt.axes([0.15, 0.20, 0.75, 0.03])
    ax_df = plt.axes([0.15, 0.15, 0.75, 0.03])
    ax_br = plt.axes([0.15, 0.10, 0.75, 0.03])
    ax_cr = plt.axes([0.15, 0.05, 0.75, 0.03])
    ax_dr = plt.axes([0.15, 0.00, 0.75, 0.03])

    bf_slider = Slider(ax_bf, "BF", 1.0, 20.0, valinit=params.BF)
    cf_slider = Slider(ax_cf, "CF", 0.5, 3.0, valinit=params.CF)
    df_slider = Slider(ax_df, "DF", 0.1, 2.0, valinit=params.DF)

    br_slider = Slider(ax_br, "BR", 1.0, 20.0, valinit=params.BR)
    cr_slider = Slider(ax_cr, "CR", 0.5, 3.0, valinit=params.CR)
    dr_slider = Slider(ax_dr, "DR", 0.1, 2.0, valinit=params.DR)

    def update(_):
        params.BF = bf_slider.val
        params.CF = cf_slider.val
        params.DF = df_slider.val
        params.BR = br_slider.val
        params.CR = cr_slider.val
        params.DR = dr_slider.val

        fy_f_new, fy_r_new = tire_curves(params, alpha_rad)

        front_line.set_ydata(fy_f_new)
        rear_line.set_ydata(fy_r_new)

        y_min = min(fy_f_new.min(), fy_r_new.min())
        y_max = max(fy_f_new.max(), fy_r_new.max())
        margin = 0.1 * max(abs(y_min), abs(y_max), 1.0)
        ax.set_ylim(y_min - margin, y_max + margin)

        fig.canvas.draw_idle()

    for slider in [
        bf_slider,
        cf_slider,
        df_slider,
        br_slider,
        cr_slider,
        dr_slider,
    ]:
        slider.on_changed(update)
        
    plt.show()
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    main()
