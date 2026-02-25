#!/usr/bin/env python3
"""
Lightweight black hole + accretion disk toy renderer.

Goals
- Runs on weak PCs
- No per-pixel star loop
- Fewer ray steps
- Small internal render, optional upscaled display

Usage
  python bh_lite.py
  python bh_lite.py --headless --output frame.ppm
  python bh_lite.py --fps 10 --scale 3
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import random
from dataclasses import dataclass


@dataclass
class Config:
    width: int = 200
    height: int = 150
    fov: float = 1.0

    gm: float = 0.13
    event_horizon: float = 0.52

    max_steps: int = 45
    ds: float = 0.14

    stars: int = 450
    bg_w: int = 640
    bg_h: int = 320

    @property
    def disk_inner(self) -> float:
        return self.event_horizon * 1.8

    @property
    def disk_outer(self) -> float:
        return 2.6


def to_ppm(width: int, height: int, rgb: bytes) -> bytes:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb


def save_ppm(path: str, width: int, height: int, rgb: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(to_ppm(width, height, rgb))


class BlackHoleLite:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = random.Random(7)
        self.bg = self._build_background()

    @staticmethod
    def _norm(x: float, y: float, z: float):
        m = math.sqrt(x * x + y * y + z * z) or 1.0
        return x / m, y / m, z / m

    @staticmethod
    def _tone_map(v: float) -> float:
        return v / (1.0 + v)

    def _build_background(self):
        cfg = self.cfg
        W, H = cfg.bg_w, cfg.bg_h

        bg = [(0.005, 0.010, 0.026)] * (W * H)

        grid_scale = 10
        for j in range(H):
            v = (j + 0.5) / H
            gv = abs((v * grid_scale) % 1.0 - 0.5)
            for i in range(W):
                u = (i + 0.5) / W
                gu = abs((u * grid_scale) % 1.0 - 0.5)
                grid = max(0.0, 1.0 - min(gu, gv) * 36.0)

                r, g, b = bg[j * W + i]
                b += 0.06 * grid
                g += 0.02 * grid
                bg[j * W + i] = (r, g, b)

        for _ in range(cfg.stars):
            sy = self.rng.uniform(-1.0, 1.0)
            a = self.rng.uniform(0.0, math.tau)
            r = math.sqrt(max(0.0, 1.0 - sy * sy))
            sx = r * math.cos(a)
            sz = r * math.sin(a)
            bright = (self.rng.random() ** 8) * 1.8 + 0.1

            u = math.atan2(sz, sx) / (2.0 * math.pi) + 0.5
            v = math.asin(max(-1.0, min(1.0, sy))) / math.pi + 0.5
            cx = int(u * W) % W
            cy = int(v * H)
            if not (0 <= cy < H):
                continue

            for dy in (-1, 0, 1):
                yy = cy + dy
                if not (0 <= yy < H):
                    continue
                for dx in (-1, 0, 1):
                    xx = (cx + dx) % W
                    fall = 1.0 / (1.0 + dx * dx + dy * dy)
                    s = bright * fall * 0.65
                    idx = yy * W + xx
                    r0, g0, b0 = bg[idx]
                    bg[idx] = (r0 + s * 0.95, g0 + s * 0.98, b0 + s * 1.08)

        return bg

    def _sample_bg_dir(self, dx: float, dy: float, dz: float):
        cfg = self.cfg
        u = math.atan2(dz, dx) / (2.0 * math.pi) + 0.5
        v = math.asin(max(-1.0, min(1.0, dy))) / math.pi + 0.5

        x = int(u * cfg.bg_w) % cfg.bg_w
        y = max(0, min(cfg.bg_h - 1, int(v * cfg.bg_h)))
        return self.bg[y * cfg.bg_w + x]

    def render_frame(self, t: float) -> bytes:
        cfg = self.cfg
        w, h = cfg.width, cfg.height
        aspect = w / h

        cam_x = math.sin(t * 0.55) * 0.45
        cam_y = 0.28 + math.sin(t * 0.33) * 0.04
        cam_z = -5.4

        rgb = bytearray(w * h * 3)
        p = 0

        for py in range(h):
            sy = 1.0 - ((py + 0.5) / h) * 2.0
            for px in range(w):
                sx = ((px + 0.5) / w) * 2.0 - 1.0

                dx, dy, dz = self._norm(sx * aspect * cfg.fov, sy * cfg.fov, 1.0)
                x, y, z = cam_x, cam_y, cam_z

                dr = dg = db = 0.0
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

                    if (old_y > 0.0 >= y) or (old_y < 0.0 <= y):
                        pr = math.hypot(x, z)
                        if cfg.disk_inner < pr < cfg.disk_outer:
                            heat = ((cfg.disk_outer - pr) / (cfg.disk_outer - cfg.disk_inner)) ** 0.6
                            tx, _, tz = self._norm(-z, 0.0, x)
                            orbital_v = math.sqrt(min(0.32, cfg.gm / (pr + 1e-5)))
                            doppler = 1.0 + orbital_v * (tx * dx + tz * dz) * 2.2
                            redshift = math.sqrt(max(0.20, 1.0 - (2.0 * cfg.gm) / (r + 1e-5)))
                            intensity = max(0.0, heat * doppler * redshift * 2.2)
                            dr += intensity * 1.25
                            dg += intensity * 0.60
                            db += intensity * 0.22

                if captured:
                    cr = cg = cb = 0.0
                else:
                    cr, cg, cb = self._sample_bg_dir(dx, dy, dz)
                    cr += dr
                    cg += dg
                    cb += db

                r8 = int((self._tone_map(cr) ** 0.85) * 255.0)
                g8 = int((self._tone_map(cg) ** 0.85) * 255.0)
                b8 = int((self._tone_map(cb) ** 0.85) * 255.0)

                rgb[p] = 0 if r8 < 0 else (255 if r8 > 255 else r8)
                rgb[p + 1] = 0 if g8 < 0 else (255 if g8 > 255 else g8)
                rgb[p + 2] = 0 if b8 < 0 else (255 if b8 > 255 else b8)
                p += 3

        return bytes(rgb)


def run_gui(renderer: BlackHoleLite, fps: int, scale: int) -> None:
    import tkinter as tk
    import tempfile
    import os

    cfg = renderer.cfg
    root = tk.Tk()
    root.title("Black Hole Lite")
    label = tk.Label(root)
    label.pack()

    state = {"frame": 0, "img": None}
    delay_ms = max(20, int(1000 / max(1, fps)))
    tmp_dir = tempfile.gettempdir()

    def tick():
        t = state["frame"] / max(1, fps)
        rgb = renderer.render_frame(t)
        ppm = to_ppm(cfg.width, cfg.height, rgb)

        # Write to a real file, then load via file=; most reliable on Windows.
        path = os.path.join(tmp_dir, f"bh_frame_{state['frame']:06d}.ppm")
        with open(path, "wb") as f:
            f.write(ppm)

        img = tk.PhotoImage(file=path)

        if scale > 1:
            img = img.zoom(scale, scale)

        label.configure(image=img)
        state["img"] = img
        state["frame"] += 1
        root.after(delay_ms, tick)

    tick()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight black hole toy renderer")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="bh_lite.ppm")
    args = parser.parse_args()

    cfg = Config(width=args.width, height=args.height)
    renderer = BlackHoleLite(cfg)

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
        run_gui(renderer, args.fps, args.scale)


if __name__ == "__main__":
    main()