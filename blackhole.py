#!/usr/bin/env python3
"""400x300 black hole + accretion disk simulation in pure Python.

Features
- Schwarzschild-inspired light bending (weak field approximation)
- Event horizon capture shadow
- Accretion disk emissive shading with Doppler + gravitational redshift cues
- Warped star field + coordinate grid for spacetime curvature perception
- Optional Tkinter real-time animation, plus headless frame export

Usage
  python blackhole_sim.py
  python blackhole_sim.py --headless --output frame.ppm
  python blackhole_sim.py --headless --frames 24 --output out/frame.ppm
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import random
from dataclasses import dataclass


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    width: int = 400
    height: int = 300
    gm: float = 0.14
    event_horizon: float = 0.56
    max_steps: int = 100
    ds: float = 0.085
    fov: float = 1.0
    stars: int = 2300

    @property
    def disk_inner(self) -> float:
        return self.event_horizon * 1.75

    @property
    def disk_outer(self) -> float:
        return 3.0


# =========================================================
# RENDERER
# =========================================================

class BlackHoleRenderer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = random.Random(7)
        self.star_points = self._generate_stars(cfg.stars)

    def _generate_stars(self, count: int):
        stars = []
        for _ in range(count):
            z = self.rng.uniform(-1.0, 1.0)
            a = self.rng.uniform(0.0, math.tau)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            x = r * math.cos(a)
            y = z
            zz = r * math.sin(a)
            brightness = self.rng.random() ** 8 * 1.8 + 0.1
            stars.append((x, y, zz, brightness))
        return stars

    @staticmethod
    def _norm(x: float, y: float, z: float):
        m = math.sqrt(x * x + y * y + z * z) or 1.0
        return x / m, y / m, z / m

    @staticmethod
    def _tone_map(v: float) -> float:
        return v / (1.0 + v)

    def _sample_background(self, dx: float, dy: float, dz: float):
        r, g, b = 0.005, 0.010, 0.026

        grid_scale = 12
        u = math.atan2(dz, dx) / (2.0 * math.pi) + 0.5
        v = math.asin(max(-1.0, min(1.0, dy))) / math.pi + 0.5
        gu = abs((u * grid_scale) % 1.0 - 0.5)
        gv = abs((v * grid_scale) % 1.0 - 0.5)
        grid = max(0.0, 1.0 - min(gu, gv) * 40.0)
        b += 0.08 * grid
        g += 0.03 * grid

        for sx, sy, sz, bright in self.star_points:
            dot = dx * sx + dy * sy + dz * sz
            if dot > 0.9994:
                hit = (dot - 0.9994) * 1900.0
                s = bright * hit
                r += s * 0.95
                g += s * 0.97
                b += s * 1.08

        return r, g, b

    # =========================================================
    # MAIN RENDER
    # =========================================================

    def render_frame(self, t: float) -> bytes:
        cfg = self.cfg
        w, h = cfg.width, cfg.height
        aspect = w / h

        cam_x = math.sin(t * 0.65) * 0.55
        cam_y = 0.31 + math.sin(t * 0.37) * 0.05
        cam_z = -6.3

        rgb = bytearray(w * h * 3)
        p = 0

        for py in range(h):
            sy = 1.0 - ((py + 0.5) / h) * 2.0
            for px in range(w):
                sx = ((px + 0.5) / w) * 2.0 - 1.0

                dx, dy, dz = self._norm(
                    sx * aspect * cfg.fov,
                    sy * cfg.fov,
                    1.0,
                )

                x, y, z = cam_x, cam_y, cam_z
                dr, dg, db = 0.0, 0.0, 0.0
                captured = False

                for _ in range(cfg.max_steps):
                    r2 = x * x + y * y + z * z
                    r = math.sqrt(r2)

                    if r < cfg.event_horizon:
                        captured = True
                        break

                    bend = (2.0 * cfg.gm) / (r2 * r + 1e-6)
                    dx += -x * bend * cfg.ds
                    dy += -y * bend * cfg.ds
                    dz += -z * bend * cfg.ds
                    dx, dy, dz = self._norm(dx, dy, dz)

                    old_y = y
                    x += dx * cfg.ds
                    y += dy * cfg.ds
                    z += dz * cfg.ds

                    # disk crossing
                    if (old_y > 0.0 >= y) or (old_y < 0.0 <= y):
                        pr = math.hypot(x, z)
                        if cfg.disk_inner < pr < cfg.disk_outer:
                            heat = (
                                (cfg.disk_outer - pr)
                                / (cfg.disk_outer - cfg.disk_inner)
                            ) ** 0.55

                            tx, _, tz = self._norm(-z, 0.0, x)
                            orbital_v = math.sqrt(
                                min(0.40, cfg.gm / (pr + 1e-5))
                            )

                            doppler = 1.0 + orbital_v * (tx * dx + tz * dz) * 2.6
                            redshift = math.sqrt(
                                max(0.14, 1.0 - (2.0 * cfg.gm) / (r + 1e-5))
                            )

                            intensity = max(
                                0.0,
                                heat * doppler * redshift * 2.8,
                            )

                            dr += intensity * 1.45
                            dg += intensity * 0.67
                            db += intensity * 0.25

                if captured:
                    cr, cg, cb = 0.0, 0.0, 0.0
                else:
                    cr, cg, cb = self._sample_background(dx, dy, dz)
                    cr += dr
                    cg += dg
                    cb += db

                rgb[p] = int((self._tone_map(cr) ** 0.82) * 255.0)
                rgb[p + 1] = int((self._tone_map(cg) ** 0.82) * 255.0)
                rgb[p + 2] = int((self._tone_map(cb) ** 0.82) * 255.0)
                p += 3

        return bytes(rgb)


# =========================================================
# PPM UTILITIES
# =========================================================

def to_ppm(width: int, height: int, rgb: bytes) -> bytes:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb


def save_ppm(path: str, width: int, height: int, rgb: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(to_ppm(width, height, rgb))


# =========================================================
# GUI
# =========================================================

def run_gui(renderer: BlackHoleRenderer, fps: int) -> None:
    import tkinter as tk

    cfg = renderer.cfg
    root = tk.Tk()
    root.title("Black Hole 3D Plane Simulation (400x300)")
    label = tk.Label(root)
    label.pack()

    state = {"frame": 0, "img": None}

    def tick():
        t = state["frame"] / fps
        rgb = renderer.render_frame(t)
        ppm = to_ppm(cfg.width, cfg.height, rgb)

        # robust transport to Tk
        b64 = base64.b64encode(ppm).decode("ascii")
        img = tk.PhotoImage(data=b64, format="PPM")

        label.configure(image=img)
        state["img"] = img
        state["frame"] += 1
        root.after(int(1000 / fps), tick)

    tick()
    root.mainloop()


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Black hole gravity + accretion disk simulator"
    )
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="blackhole_frame.ppm")
    args = parser.parse_args()

    cfg = Config(width=args.width, height=args.height)
    renderer = BlackHoleRenderer(cfg)

    if args.headless:
        for i in range(args.frames):
            rgb = renderer.render_frame(i / max(1, args.fps))

            if args.frames == 1:
                out = args.output
            else:
                base, ext = os.path.splitext(args.output)
                ext = ext or ".ppm"
                out = f"{base}_{i:04d}{ext}"

            save_ppm(out, cfg.width, cfg.height, rgb)
            print(f"wrote {out}")
    else:
        run_gui(renderer, args.fps)


if __name__ == "__main__":
    main()