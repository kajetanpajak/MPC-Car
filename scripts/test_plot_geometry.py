import matplotlib.pyplot as plt
import numpy as np

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.track.geometry import curvilinear_to_global
from racing_sim.track.circle import CircleTrack

def main():
    track = CircleTrack(2.0, 1.0, 1.0)
    vehicle_state = VehicleState(s=(np.pi - 1.0) * track.radius,
                                 n=-0.5,
                                 mu=np.pi / 4,
                                 vx=0.0,
                                 vy=0.0,
                                 r=0.0,
                                 delta=0.0,
                                 T=0.0)
    
    # track_curvature = [track.curvature(s) for s in np.arange(0.0, 2*np.pi, 0.05)]
    track_coords = np.array([track.position(s) for s in np.arange(0.0, 2*np.pi * track.radius, 0.05)])
    
    car_coords = curvilinear_to_global(vehicle_state, track)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    heading_vector_length = 0.5
    heading_dx = heading_vector_length * np.cos(car_coords.psi)
    heading_dy = heading_vector_length * np.sin(car_coords.psi)
    
    ax.plot(track_coords[:, 0], track_coords[:, 1], label="track centerline")
    ax.quiver(
        car_coords.x,
        car_coords.y,
        heading_dx,
        heading_dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="r",
        width=0.008,
        label="vehicle heading",
    )
    ax.scatter([car_coords.x], [car_coords.y], c="r", s=40)
    ax.grid()
    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
