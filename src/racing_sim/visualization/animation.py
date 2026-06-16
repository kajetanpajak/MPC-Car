from dataclasses import dataclass

import pygame
import numpy as np

from racing_sim.model.params import VehicleParams
from racing_sim.model.state import VehicleState
from racing_sim.track.geometry import curvilinear_to_global
from racing_sim.track.tracks import CircleTrack

BACKGROUND = (28, 34, 40)
GRID = (42, 49, 56)
ASPHALT = (58, 63, 68)
TRACK_EDGE = (225, 225, 215)
CENTERLINE = (245, 205, 85)
PATH = (82, 190, 255)
CAR_BODY = (235, 72, 62)
CAR_ACCENT = (255, 195, 92)
CAR_OUTLINE = (18, 22, 26)
WHEEL = (12, 14, 16)
TEXT = (235, 238, 240)
TEXT_DIM = (165, 174, 184)
PANEL = (18, 23, 28)

@dataclass
class Frame:
    time: float
    state: VehicleState
    x: float
    y: float
    psi: float
    delta_dot_cmd: float = 0.0
    T_dot_cmd: float = 0.0

class RaceReplay:
    def __init__(
        self,
        frames: list[Frame],
        track,
        params: VehicleParams,
        width: int=1000,
        height: int=800,
        scale: float=15.0):
        
        self.frames = frames
        self.track = track
        self.params = params
        
        self.width = width
        self.height = height
        self.scale = scale
        
        self.offset_x = width // 2
        self.offset_y = height // 2
        
        self.frame_id = 0
        self.paused = False
        
        self.playback_speed = 1
        self._center_camera()
        
    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        screen_x = self.offset_x + x * self.scale
        screen_y = self.offset_y + y * self.scale
        return (int(screen_x), int(screen_y))

    def _track_s_values(self, count: int = 1000):
        if hasattr(self.track, "length"):
            return np.linspace(0.0, self.track.length, count)

        s_max = max(frame.state.s for frame in self.frames)
        return np.linspace(0.0, s_max, count)

    def _center_camera(self):
        if not self.frames:
            return

        xs = [frame.x for frame in self.frames]
        ys = [frame.y for frame in self.frames]

        for s in self._track_s_values(250):
            x, y = self.track.position(s)
            xs.append(x)
            ys.append(y)

        center_x = 0.5 * (min(xs) + max(xs))
        center_y = 0.5 * (min(ys) + max(ys))
        self.offset_x = self.width // 2 - center_x * self.scale
        self.offset_y = self.height // 2 - center_y * self.scale

    def draw_grid(self, screen):
        spacing = max(int(5.0 * self.scale), 20)

        start_x = int(self.offset_x) % spacing
        start_y = int(self.offset_y) % spacing

        for x in range(start_x, self.width, spacing):
            pygame.draw.line(screen, GRID, (x, 0), (x, self.height), 1)

        for y in range(start_y, self.height, spacing):
            pygame.draw.line(screen, GRID, (0, y), (self.width, y), 1)
    
    def draw_track(self, screen):
        center_points = []
        left_points = []
        right_points = []
        
        for s in self._track_s_values():
            x, y = self.track.position(s)
            center_points.append(self.world_to_screen(x, y))

            if hasattr(self.track, "heading"):
                psi = self.track.heading(s)
                nx = -np.sin(psi)
                ny = np.cos(psi)
                left_points.append(
                    self.world_to_screen(
                        x + self.track.width_left * nx,
                        y + self.track.width_left * ny,
                    )
                )
                right_points.append(
                    self.world_to_screen(
                        x - self.track.width_right * nx,
                        y - self.track.width_right * ny,
                    )
                )
            
        closed = hasattr(self.track, "length")

        if len(left_points) >= 2 and len(right_points) >= 2:
            road_polygon = left_points + list(reversed(right_points))
            pygame.draw.polygon(screen, ASPHALT, road_polygon)
            pygame.draw.lines(screen, TRACK_EDGE, closed, left_points, 2)
            pygame.draw.lines(screen, TRACK_EDGE, closed, right_points, 2)

        if len(center_points) >= 2:
            pygame.draw.lines(screen, CENTERLINE, closed, center_points, 1)
        
    def draw_path(self, screen):
        points = [
            self.world_to_screen(frame.x, frame.y)
            for frame in self.frames[: self.frame_id + 1]
        ]
        
        if len(points) >= 2:
            pygame.draw.lines(screen, PATH, False, points, 3)
            
    def draw_car(self, screen, frame: Frame):
        length = self.params.Lc * self.scale
        width = self.params.Wc * self.scale
        
        cx, cy = self.world_to_screen(frame.x, frame.y)
        
        corners = np.array([
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ])
        
        psi = frame.psi
        
        rot = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi), np.cos(psi)]
        ])
        
        rotated = corners @ rot.T
        
        screen_points = [
            (int(cx + px), int(cy + py))
            for px, py in rotated
        ]
        
        shadow_points = [(x + 3, y + 4) for x, y in screen_points]
        pygame.draw.polygon(screen, (10, 12, 14), shadow_points)
        pygame.draw.polygon(screen, CAR_BODY, screen_points)
        pygame.draw.polygon(screen, CAR_OUTLINE, screen_points, 2)

        windshield = np.array([
            [length * 0.18, width * 0.34],
            [length * 0.38, width * 0.18],
            [length * 0.38, -width * 0.18],
            [length * 0.18, -width * 0.34],
        ])
        windshield_points = [
            (int(cx + px), int(cy + py))
            for px, py in windshield @ rot.T
        ]
        pygame.draw.polygon(screen, CAR_ACCENT, windshield_points)

        wheel_length = max(length * 0.22, 8)
        wheel_width = max(width * 0.16, 4)
        wheel_centers = [
            [length * 0.28, width * 0.56],
            [length * 0.28, -width * 0.56],
            [-length * 0.28, width * 0.56],
            [-length * 0.28, -width * 0.56],
        ]

        for wx, wy in wheel_centers:
            wheel_corners = np.array([
                [wx + wheel_length / 2, wy + wheel_width / 2],
                [wx + wheel_length / 2, wy - wheel_width / 2],
                [wx - wheel_length / 2, wy - wheel_width / 2],
                [wx - wheel_length / 2, wy + wheel_width / 2],
            ])
            wheel_points = [
                (int(cx + px), int(cy + py))
                for px, py in wheel_corners @ rot.T
            ]
            pygame.draw.polygon(screen, WHEEL, wheel_points)

        nose_x = cx + np.cos(psi) * length / 2
        nose_y = cy + np.sin(psi) * length / 2
        pygame.draw.line(screen, CAR_OUTLINE, (cx, cy), (nose_x, nose_y), 3)
        
    def draw_text(self, screen, font, frame: Frame):
        state = frame.state
        
        lines = [
            f"time: {frame.time:.2f} s",
            f"frame: {self.frame_id}/{len(self.frames) - 1}",
            f"vx: {state.vx:.2f} m/s",
            f"n: {state.n:.2f} m",
            f"mu: {state.mu:.2f} rad",
            f"delta: {state.delta:.2f} rad",
            f"T: {state.T:.2f}",
            f"speed: x{self.playback_speed}",
            "space pause | arrows step/speed | r restart | esc quit",
        ]

        panel_width = 310
        panel_height = 24 + 22 * len(lines)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((*PANEL, 215))
        screen.blit(panel, (16, 16))

        x = 30
        y = 28

        for i, line in enumerate(lines):
            color = TEXT if i < len(lines) - 1 else TEXT_DIM
            surface = font.render(line, True, color)
            screen.blit(surface, (x, y))
            y += 22
            
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False

            if event.key == pygame.K_SPACE:
                self.paused = not self.paused

            elif event.key == pygame.K_r:
                self.frame_id = 0

            elif event.key == pygame.K_RIGHT:
                self.frame_id = min(self.frame_id + 1, len(self.frames) - 1)

            elif event.key == pygame.K_LEFT:
                self.frame_id = max(self.frame_id - 1, 0)

            elif event.key == pygame.K_UP:
                self.playback_speed = min(self.playback_speed + 1, 10)

            elif event.key == pygame.K_DOWN:
                self.playback_speed = max(self.playback_speed - 1, 1)

        return True
    
    def run(self):
        pygame.init()
        
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Trajectory replay")
        
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)
        
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    running = self.handle_event(event)
                    
            if not self.paused:
                self.frame_id += self.playback_speed
                
                if self.frame_id >= len(self.frames):
                    self.frame_id = len(self.frames) - 1
                    self.paused = True
                    
            frame = self.frames[self.frame_id]
            
            screen.fill(BACKGROUND)
            
            self.draw_grid(screen)
            self.draw_track(screen)
            self.draw_path(screen)
            self.draw_car(screen, frame)
            self.draw_text(screen, font, frame)
            
            pygame.display.flip()
            clock.tick(30)
            
        pygame.quit()
        
def make_fake_frames(track, params) -> list[Frame]:
    """
    Temporary demo trajectory.
    Replace this with your MPC/simulation-generated states.
    """
    frames = []

    dt = 0.05
    num_frames = 300

    for i in range(num_frames):
        t = i * dt
        s = 5.0 * t

        state = VehicleState(
            s=s,
            n=0.0,
            mu=0.0,
            vx=5.0,
            vy=0.0,
            r=0.0,
            delta=0.0,
            T=0.0,
        )

        pose = curvilinear_to_global(state, track)

        frames.append(
            Frame(
                time=t,
                state=state,
                x=pose.x,
                y=pose.y,
                psi=pose.psi,
            )
        )

    return frames


def main():
    params = VehicleParams.default()

    track = CircleTrack(
        radius=20.0,
        width_left=1.0,
        width_right=1.0,
    )

    frames = make_fake_frames(track, params)

    replay = RaceReplay(
        frames=frames,
        track=track,
        params=params,
        width=1000,
        height=800,
        scale=15.0,
    )

    replay.run()


if __name__ == "__main__":
    main()
