from dataclasses import dataclass
import numpy as np

@dataclass
class CircleTrack:
    radius: float
    width_left: float
    width_right: float
    
    def curvature(self, s: float) -> float:
        """
        :param s: Progress - arc-length along the track (not used, keep for consistency).
        :type s: float
        :return: Curvature at progress s.
        :rtype: float
        """
        return 1.0 / self.radius
    
    def position(self, s: float) -> tuple[float, float]:
        """
        :param s: Progress - arc-length along the track.
        :type s: float
        :return: Coordinates of the track track at progress s (x, y).
        :rtype: tuple[float, float]
        """
        theta = s / self.radius
        x = self.radius * np.sin(theta)
        y = self.radius * (1.0 - np.cos(theta))
        return (x, y)
    
    def heading(self, s: float) -> float:
        """    
        :param s: Progress - arc-length along the track.
        :type s: float
        :return: Heading - angle of the track's tangent at progress s (rad).
        :rtype: float
        """
        return s / self.radius

@dataclass
class StraightTrack:
    width_left: float
    width_right: float
    
    def curvature(self, s: float) -> float:
        return 0.0
    
    def position(self, s: float) -> float:
        return s, 0.0
    
    def heading(self, s: float) -> float:
        return 0.0
    
class EllipseTrack:
    def __init__(self,
                 a: float,
                 b: float,
                 width_left: float,
                 width_right:float,
                 num_samples: int = 2_000):
        self.a = a
        self.b = b
        self.width_left = width_left
        self.width_right = width_right
        self.num_samples = num_samples
        
        self.sample_track()
        
    
    def sample_track(self):
        
        theta = np.linspace(0.0, 2 * np.pi, self.num_samples, endpoint=False)
        
        xs = self.a * np.cos(theta)
        ys = self.b * np.sin(theta)
        
        xs_temp = np.append(xs, xs[0])
        ys_temp = np.append(ys, ys[0])
        
        dxs = np.diff(xs_temp)
        dys = np.diff(ys_temp)
        
        dss = np.sqrt(dxs ** 2 + dys ** 2)
        
        self.s_samples = np.concatenate(([0.0], np.cumsum(dss)))
        self.length = self.s_samples[-1]
        
        self.x_samples = xs_temp
        self.y_samples = ys_temp
        self.theta_samples = np.append(theta, [2*np.pi])
        
        dx_dtheta = - self.a * np.sin(self.theta_samples)
        dy_dtheta = self.b * np.cos(self.theta_samples)
        
        self.psi_samples = np.unwrap(np.arctan2(dy_dtheta, dx_dtheta)) # heading samples
        
        dx_ddtheta = -self.a * np.cos(self.theta_samples)
        dy_ddtheta = -self.b * np.sin(self.theta_samples)
        
        self.kappa_samples = (   (dx_dtheta * dy_ddtheta - dy_dtheta * dx_ddtheta) / 
                                 (dx_dtheta ** 2 + dy_dtheta ** 2) ** 1.5    )
        
        
    def curvature(self, s: float) -> float:
        s_wrapped = s % self.length
        
        kappa = np.interp(s_wrapped, self.s_samples, self.kappa_samples)
        return float(kappa)
        
    def position(self, s: float) -> tuple[float, float]:
        
        s_wrapped = s % self.length
        
        x = np.interp(s_wrapped, self.s_samples, self.x_samples)
        y = np.interp(s_wrapped, self.s_samples, self.y_samples)
        
        return (float(x), float(y))
    
    def heading(self, s: float) -> float:
        s_wrapped = s % self.length
        psi = np.interp(s_wrapped, self.s_samples, self.psi_samples)
        
        psi_wrapped = (psi + np.pi) % (2.0 * np.pi) - np.pi
        
        return float(psi_wrapped)
    
    
from scipy.interpolate import CubicSpline
class WaypointTrack:
    def __init__(
        self,
        points: np.ndarray,
        width_left: float,
        width_right: float,
        closed: bool=True):
        self.points = points
        self.width_left = width_left
        self.width_right = width_right
        self.closed = closed
        
        self.create_splines()
    
    def create_splines(self):
        dx = np.diff(self.points[:, 0])
        dy = np.diff(self.points[:, 1])
        
        ds = np.sqrt(dx ** 2 + dy ** 2)
        
        self.s_samples = np.concatenate(([0.0], np.cumsum(ds)))
        self.length = self.s_samples[-1]
        
        bc_type = "not-a-knot"
        if self.closed:
            bc_type = "periodic"
        
        self.x_spline = CubicSpline(self.s_samples, self.points[:, 0], bc_type=bc_type)
        self.y_spline = CubicSpline(self.s_samples, self.points[:, 1], bc_type=bc_type)
        
    
    def curvature(self, s: float):
        if self.closed:
            s = s % self.length
        
        kappa = ((self.x_spline(s, 1) * self.y_spline(s, 2) - self.x_spline(s, 2) * self.y_spline(s, 2)) /
                 (self.x_spline(s, 1) ** 2 + self.y_spline(s, 1) ** 2) ** 1.5)
        
        return kappa
        
        
    def position(self, s: float) -> tuple[float, float]:
        if self.closed:
            s = s % self.length
            
        x = self.x_spline(s)
        y = self.y_spline(s)
        
        return (float(x), float(y))    
    
    def heading(self, s: float):
        if self.closed:
            s = s % self.length
        dx_s = self.x_spline(s, 1)
        dy_s = self.y_spline(s, 1)
        
        return np.atan2(dy_s, dx_s)
    
if __name__ == "__main__":
    pts = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [15.0, 5.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])
    
    track = WaypointTrack(pts, 1.0, 1.0, True)
    print(track.length)
    
    s_range = np.arange(0.0, 50.0, 0.2)
    xs = []
    ys = []
    
    for s in s_range:
        x, y = track.position(s)
        
        xs.append(x)
        ys.append(y)
        
    import matplotlib.pyplot as plt
    
    plt.plot(xs, ys)
    plt.grid()
    plt.show()
        
    
    