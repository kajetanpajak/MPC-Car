from dataclasses import dataclass

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
