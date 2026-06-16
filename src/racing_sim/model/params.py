from dataclasses import dataclass


@dataclass
class VehicleParams:
    # Vehicle inertia and geometry
    m: float
    Iz: float
    lF: float
    lR: float
    Lc: float
    Wc: float

    # Simplified Pacejka tire parameters
    BF: float
    BR: float
    CF: float
    CR: float
    DF: float
    DR: float

    # Longitudinal force model
    Cm: float
    Cr0: float
    Cr2: float

    # Torque-vectoring yaw moment gain
    ptv: float
    
    # Actuator limits
    delta_min: float
    delta_max: float
    delta_dot_min: float
    delta_dot_max: float
    T_min: float
    T_max: float
    T_dot_min: float
    T_dot_max: float

    # Friction-ellipse parameters ??????????
    rho_long: float
    lambda_front: float
    lambda_rear: float

    # Numerical safety min velocity
    min_vx: float

    @classmethod
    def default(cls) -> "VehicleParams":
        return cls(
            m=230.0,
            Iz=110.0,
            lF=0.95,
            lR=0.85,
            Lc=1.80,
            Wc=1.20,
            BF=9.0,
            BR=9.5,
            CF=1.3,
            CR=1.3,
            DF=1.0,
            DR=1.0,
            Cm=4200.0,
            Cr0=180.0,
            Cr2=0.65,
            ptv=40.0,
            delta_min=-0.50,
            delta_max=0.50,
            delta_dot_min=-0.80,
            delta_dot_max=0.80,
            T_min=-0.7,
            T_max=1.0,
            T_dot_min=-2.0,
            T_dot_max=1.5,
            rho_long=0.90,
            lambda_front=0.95,
            lambda_rear=0.95,
            min_vx=0.50,
        )


if __name__ == "__main__":
    params = VehicleParams.default()  
    print(params.m)  
    
