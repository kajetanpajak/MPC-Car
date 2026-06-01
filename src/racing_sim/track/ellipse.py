# from dataclasses import dataclass
import numpy as np

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
        
        
        
def main():
    track = EllipseTrack(10.0, 5.0, 1.0, 1.0)
    import matplotlib.pyplot as plt
    
    print(f"Length: {track.length}")
    s = 48.4 + 4.0
    
    x, y = track.position(s)
    print(f"Heading: {np.rad2deg(track.heading(s))}")
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 6))
    
    ax.plot(track.x_samples, track.y_samples)
    ax.scatter([x], [y], s=40, c='r')
    ax.grid()
    plt.show()
    
    
if __name__ == "__main__":
    main()