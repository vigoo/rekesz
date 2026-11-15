#!/usr/bin/env python3
"""Regenerate the Rekesz app icon from its parametric design.

The icon is a rainbow film strip sweeping diagonally across a dark rounded
plate; each frame holds a tiny landscape (a sun and two hill layers). This
script writes the SVG master and every raster derived from it:

  assets/icon/rekesz.svg           master artwork
  assets/icon/png/rekesz-<N>.png   icon-theme sizes, installed by
                                   scripts/install-desktop-entry.sh
  assets/icon/rekesz.ico           Windows icon, embedded by build.rs

Rasters come from librsvg (`rsvg-convert`); the .ico is assembled with
ImageMagick (`magick`). Icon themes get the PNGs rather than the SVG because
KDE's Qt SVG renderer ignores the plate's clip path.
"""

import bisect
import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
ICON_DIR = ROOT / "assets" / "icon"
THEME_SIZES = (16, 22, 24, 32, 48, 64, 96, 128, 256, 512)
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

S = 512
C = S / 2

# Rounded-square plate, as path data.
PLATE = (
    "M146 32 H366 A114 114 0 0 1 480 146 V366 A114 114 0 0 1 366 480 "
    "H146 A114 114 0 0 1 32 366 V146 A114 114 0 0 1 146 32 Z"
)


def svg(defs, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}">\n'
        f'<defs>{defs}<clipPath id="clip"><path d="{PLATE}"/></clipPath></defs>\n'
        f'<g clip-path="url(#clip)">{body}</g>\n'
        # Hairline glass edge on top of everything.
        f'<path d="{PLATE}" fill="none" stroke="#FFFFFF" stroke-opacity="0.10" stroke-width="3"/>\n'
        "</svg>\n"
    )


def plate():
    return f'<path d="{PLATE}" fill="url(#bg)"/><path d="{PLATE}" fill="url(#hl)"/>'


def highlight(cx, cy, r, opacity):
    return (
        f'<radialGradient id="hl" cx="{cx}" cy="{cy}" r="{r}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="{opacity}"/>'
        f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>'
        "</radialGradient>"
    )


def drop_shadow(dy, blur, opacity):
    return (
        f'<filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feGaussianBlur in="SourceAlpha" stdDeviation="{blur}"/>'
        f'<feOffset dy="{dy}" result="b"/>'
        f'<feComponentTransfer><feFuncA type="linear" slope="{opacity}"/></feComponentTransfer>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
    )


def bez(p, t):
    mt = 1 - t
    return tuple(
        mt**3 * a + 3 * mt * mt * t * b + 3 * mt * t * t * c + t**3 * d
        for a, b, c, d in zip(*p)
    )


def bez_d(p, t):
    mt = 1 - t
    return tuple(
        3 * mt * mt * (b - a) + 6 * mt * t * (c - b) + 3 * t * t * (d - c)
        for a, b, c, d in zip(*p)
    )


def offset_path(p, off, n=240):
    """Polyline following the curve, shifted `off` along its normal."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x, y = bez(p, t)
        dx, dy = bez_d(p, t)
        length = math.hypot(dx, dy)
        pts.append((x - off * dy / length, y + off * dx / length))
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)


class Curve:
    """Arc-length parametrisation of a cubic Bezier, for placing shapes along it."""

    def __init__(self, p, n=4000):
        self.pts, self.ders, self.cum = [], [], []
        for i in range(n + 1):
            t = i / n
            pt = bez(p, t)
            self.cum.append(self.cum[-1] + math.dist(pt, self.pts[-1]) if self.pts else 0.0)
            self.pts.append(pt)
            self.ders.append(bez_d(p, t))
        self.length = self.cum[-1]

    def at(self, s, v=0.0):
        """Point at arc length s, offset v along the normal (same side as offset_path)."""
        i = min(max(bisect.bisect_left(self.cum, s), 1), len(self.cum) - 1)
        s0, s1 = self.cum[i - 1], self.cum[i]
        f = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
        (x0, y0), (x1, y1) = self.pts[i - 1], self.pts[i]
        dx, dy = self.ders[i]
        length = math.hypot(dx, dy)
        x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        return (x - v * dy / length, y + v * dx / length)

    def nearest(self, q):
        i = min(range(len(self.pts)), key=lambda j: math.dist(self.pts[j], q))
        return self.cum[i]


def poly(points):
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"


def hills(base, bumps):
    """Rounded ridge: V(u) for u in [0, 1]; V = -1 is the sky edge, +1 the ground edge."""
    return lambda u: base - sum(a * math.exp(-(((u - c) / w) ** 2)) for c, w, a in bumps)


def peaks(base, tops):
    """Mountain ridge made of straight-sided peaks."""
    return lambda u: base - max([0.0] + [a * max(0.0, 1 - abs(u - c) / w) for c, w, a in tops])


# Per-frame scenes: (sun u, sun v, far ridge, near ridge), cycled along the strip.
SCENES = [
    (0.70, -0.42, hills(0.30, [(0.25, 0.20, 0.55), (0.75, 0.18, 0.30)]),
     hills(0.66, [(0.58, 0.26, 0.40)])),
    (0.28, -0.40, peaks(0.40, [(0.62, 0.30, 0.95), (0.30, 0.22, 0.55)]),
     hills(0.72, [(0.30, 0.30, 0.30)])),
    (0.62, -0.34, hills(0.22, [(0.40, 0.30, 0.35)]),
     hills(0.60, [(0.80, 0.22, 0.42), (0.15, 0.20, 0.30)])),
]


def scene_frames(p, frame_len, gap, half_h, steps=28):
    """Curved film frames, each holding a tiny landscape (sun + two hill layers)."""
    curve = Curve(p)
    centre_s = curve.nearest((C, C))
    pitch = frame_len + gap
    out = []
    for j in range(-5, 6):
        s0 = centre_s - frame_len / 2 + j * pitch
        if s0 < 0 or s0 + frame_len > curve.length:
            continue
        sun_u, sun_v, far, near = SCENES[j % len(SCENES)]

        def uv(u, v, s0=s0):
            return curve.at(s0 + u * frame_len, v * half_h)

        us = [i / steps for i in range(steps + 1)]
        ground = [uv(u, 1) for u in reversed(us)]
        frame = [uv(u, -1) for u in us] + ground
        far_ridge = [uv(u, far(u)) for u in us] + ground
        near_ridge = [uv(u, near(u)) for u in us] + ground
        sx, sy = uv(sun_u, sun_v)
        cid = f"frame{j + 5}"
        out.append(
            f'<clipPath id="{cid}"><path d="{poly(frame)}"/></clipPath>'
            f'<path d="{poly(frame)}" fill="url(#frames)"/>'
            f'<g clip-path="url(#{cid})">'
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{half_h * 0.44:.1f}" fill="#FFFFFF" opacity="0.2"/>'
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{half_h * 0.21:.1f}" fill="#FFF8E7" opacity="0.95"/>'
            f'<path d="{poly(far_ridge)}" fill="#2E1065" opacity="0.30"/>'
            f'<path d="{poly(near_ridge)}" fill="#2E1065" opacity="0.52"/>'
            "</g>"
        )
    return "\n".join(out)


def icon_svg():
    # A rising diagonal S that fills the plate corner to corner.
    p = ((-70, 520), (150, 200), (362, 420), (590, -10))
    band = 182
    centre = offset_path(p, 0)
    defs = f"""
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#151C33"/><stop offset="1" stop-color="#070B16"/>
</linearGradient>
{highlight(160, 110, 340, 0.14)}
<linearGradient id="strip" x1="0" y1="0" x2="0" y2="512" gradientUnits="userSpaceOnUse">
  <stop offset="0" stop-color="#2A3147"/><stop offset="1" stop-color="#151A2B"/>
</linearGradient>
<linearGradient id="frames" x1="70" y1="450" x2="442" y2="62" gradientUnits="userSpaceOnUse">
  <stop offset="0" stop-color="#FBBF24"/>
  <stop offset="0.28" stop-color="#FB7185"/>
  <stop offset="0.52" stop-color="#C084FC"/>
  <stop offset="0.76" stop-color="#60A5FA"/>
  <stop offset="1" stop-color="#2DD4BF"/>
</linearGradient>
{drop_shadow(dy=10, blur=16, opacity=0.6)}
"""
    hole_off = band / 2 - 21
    frames = scene_frames(p, frame_len=96, gap=14, half_h=(band - 72) / 2)
    body = f"""
{plate()}
<g filter="url(#shadow)">
  <path d="{centre}" fill="none" stroke="url(#strip)" stroke-width="{band}"/>
</g>
{frames}
<g fill="none" stroke="#EEF2F7" stroke-width="15" stroke-linecap="round"
   stroke-dasharray="8 24">
  <path d="{offset_path(p, hole_off)}"/>
  <path d="{offset_path(p, -hole_off)}"/>
</g>
"""
    return svg(defs, body)


def render(svg_path, size, out_path):
    subprocess.run(
        ["rsvg-convert", "-w", str(size), "-h", str(size), str(svg_path), "-o", str(out_path)],
        check=True,
    )


def main():
    master = ICON_DIR / "rekesz.svg"
    master.write_text(icon_svg())
    print("wrote", master.relative_to(ROOT))

    png_dir = ICON_DIR / "png"
    png_dir.mkdir(exist_ok=True)
    for size in THEME_SIZES:
        out = png_dir / f"rekesz-{size}.png"
        render(master, size, out)
        print("wrote", out.relative_to(ROOT))

    with tempfile.TemporaryDirectory() as scratch:
        frames = []
        for size in ICO_SIZES:
            frame = Path(scratch) / f"{size}.png"
            render(master, size, frame)
            frames.append(str(frame))
        ico = ICON_DIR / "rekesz.ico"
        subprocess.run(["magick", *frames, str(ico)], check=True)
        print("wrote", ico.relative_to(ROOT))


if __name__ == "__main__":
    main()
