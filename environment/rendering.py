"""OpenGL human renderer and deterministic report-frame renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from environment.custom_env import AquaForecastEnv


WIDTH, HEIGHT = 1120, 680
NAVY = (13, 35, 58)
BLUE = (24, 126, 178)
CYAN = (78, 205, 196)
PALE = (232, 245, 247)
GREEN = (38, 166, 91)
ORANGE = (239, 143, 54)
RED = (211, 74, 74)
WHITE = (248, 250, 252)
MUTED = (126, 148, 165)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _metric_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, value: str, accent: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=14, fill=(25, 51, 76), outline=(51, 79, 103), width=2)
    draw.rectangle((x0, y0, x0 + 7, y1), fill=accent)
    draw.text((x0 + 20, y0 + 14), title.upper(), font=_font(13, True), fill=MUTED)
    draw.text((x0 + 20, y0 + 40), value, font=_font(25, True), fill=WHITE)


def draw_rgb_frame(env: "AquaForecastEnv") -> np.ndarray:
    """Create a high-resolution dashboard frame for reports/video export."""
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 78), fill=(10, 28, 47))
    draw.text((34, 18), "AquaForecast", font=_font(30, True), fill=WHITE)
    draw.text((300, 26), "Mission-Based Water Operations Simulator", font=_font(18), fill=CYAN)
    draw.text((885, 25), f"DAY {env.day:02d}/{env.episode_days}", font=_font(18, True), fill=WHITE)

    # Reservoir and animated-looking water layers.
    rx0, ry0, rx1, ry1 = 55, 145, 405, 565
    draw.rounded_rectangle((rx0, ry0, rx1, ry1), radius=28, fill=(19, 47, 72), outline=(112, 151, 176), width=4)
    fraction = float(np.clip(env.reservoir_ml / env.CAPACITY_ML, 0.0, 1.0))
    water_top = int(ry1 - 18 - fraction * (ry1 - ry0 - 36))
    draw.rounded_rectangle((rx0 + 18, water_top, rx1 - 18, ry1 - 18), radius=16, fill=BLUE)
    for offset in range(0, 330, 28):
        y = water_top + 10 + (offset // 28 % 2) * 5
        draw.arc((rx0 + 20 + offset, y - 8, rx0 + 70 + offset, y + 8), 190, 350, fill=CYAN, width=3)
    draw.text((82, 97), "COMMUNITY RESERVOIR", font=_font(17, True), fill=WHITE)
    draw.text((145, 310), f"{fraction * 100:0.1f}%", font=_font(48, True), fill=WHITE)
    draw.text((138, 390), f"{env.reservoir_ml:0.2f} / {env.CAPACITY_ML:.0f} ML", font=_font(17), fill=PALE)

    # Flow path and service node.
    draw.line((405, 355, 510, 355), fill=CYAN, width=14)
    for x in (430, 465, 500):
        draw.polygon([(x, 342), (x + 18, 355), (x, 368)], fill=WHITE)
    draw.ellipse((490, 290, 620, 420), fill=(26, 70, 95), outline=CYAN, width=4)
    draw.text((520, 318), "OPS", font=_font(24, True), fill=WHITE)
    draw.text((507, 357), "AGENT", font=_font(18, True), fill=CYAN)
    draw.line((620, 355, 700, 355), fill=CYAN, width=14)
    draw.polygon([(665, 342), (685, 355), (665, 368)], fill=WHITE)

    # Metric cards.
    _metric_card(draw, (710, 112, 1080, 195), "Forecast demand", f"{env.forecast_demand_ml:.3f} ML/day", BLUE)
    _metric_card(draw, (710, 214, 1080, 297), "Weather", f"{env.temperature_c:.1f} C  |  {env.rainfall_mm:.1f} mm", CYAN)
    _metric_card(draw, (710, 316, 1080, 399), "Leakage risk", f"{env.leakage_rate * 100:.1f}%", ORANGE if env.leakage_rate > 0.08 else GREEN)
    _metric_card(draw, (710, 418, 1080, 501), "Treatment health", f"{env.treatment_health * 100:.1f}%", GREEN if env.treatment_health > 0.75 else RED)

    # Recommendation strip.
    draw.rounded_rectangle((55, 595, 1080, 650), radius=14, fill=(20, 58, 83), outline=(55, 96, 121), width=2)
    draw.text((75, 607), "RECOMMENDATION", font=_font(13, True), fill=CYAN)
    action = env.ACTIONS[env._last_action].name
    draw.text((250, 604), action, font=_font(20, True), fill=WHITE)
    draw.text((875, 608), f"Reward {env._last_reward:+.2f}", font=_font(16, True), fill=WHITE)
    return np.asarray(image, dtype=np.uint8)


class AquaForecastRenderer:
    """Pyglet renderer; pyglet draws the dashboard through an OpenGL context."""

    def __init__(self, env: "AquaForecastEnv") -> None:
        import pyglet

        self.pyglet = pyglet
        self.window = pyglet.window.Window(WIDTH, HEIGHT, "AquaForecast RL Simulator", resizable=False, vsync=True)
        self.sprite = None
        self.window.push_handlers(on_close=self._on_close)

    def _on_close(self) -> None:
        self.window.close()

    def draw(self, env: "AquaForecastEnv") -> None:
        if self.window.has_exit:
            return
        frame = draw_rgb_frame(env)
        image = self.pyglet.image.ImageData(WIDTH, HEIGHT, "RGB", frame.tobytes(), pitch=-WIDTH * 3)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()
        image.blit(0, 0)
        self.window.flip()

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
