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
