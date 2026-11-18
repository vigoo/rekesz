#!/usr/bin/env python3
"""Seed a scratch photo library for the screenshot tour's scratch suite.

Creates deterministic, colorful synthetic JPEGs (no real content, no EXIF —
the scratch captures deliberately have no GPS data) laid out as:

  <library>/                     a few images in the root
  <library>/People/              face-analysis fixture images from the repo
  <library>/Empty Album/Day 1/   no images at all (empty states)
  <source>/                      larger JPEGs to import (slow enough to capture
                                 the progress footer mid-run)
  <destination>/                 empty import destination

Usage: seed-shot-library.py <library> <source> <destination> [--repo-root R]
"""

import argparse
import math
import random
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent


def palette(index: int) -> list:
    random.seed(index)
    base = (
        random.randint(40, 200),
        random.randint(40, 200),
        random.randint(40, 200),
    )
    accent = (
        (base[0] + 120) % 256,
        (base[1] + 60) % 256,
        (base[2] + 180) % 256,
    )
    return base, accent


def make_photo(path: Path, width: int, height: int, index: int, noisy: bool) -> None:
    """Write one synthetic photo: gradient sky, sun disc, hill silhouette,
    and optional per-pixel noise (noise makes the JPEG large and slow to copy
    so the tour can capture the import progress footer mid-run)."""
    from PIL import Image, ImageDraw

    base, accent = palette(index)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(0, width, 4):
            t = y / height
            shade = tuple(int(base[i] + (accent[i] - base[i]) * t) for i in range(3))
            for dx in range(4):
                if x + dx < width:
                    pixels[x + dx, y] = shade
    draw = ImageDraw.Draw(image)
    sun_x = width // 3 + (index * 97) % (width // 3)
    sun_y = height // 4
    radius = min(width, height) // 8
    draw.ellipse(
        (sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius),
        fill=(255, 244, 214),
    )
    hill_height = height * 2 // 3
    points = []
    for x in range(0, width + 1, max(1, width // 32)):
        offset = math.sin(x / width * math.pi * 2 + index) * height // 12
        points.append((x, hill_height + offset))
    points += [(width, height), (0, height)]
    draw.polygon(points, fill=tuple(int(c * 0.45) for c in base))
    if noisy:
        rng = random.Random(index)
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                jitter = rng.randint(-28, 28)
                pixels[x, y] = (
                    max(0, min(255, r + jitter)),
                    max(0, min(255, g + jitter)),
                    max(0, min(255, b + jitter)),
                )
    image.save(path, "JPEG", quality=95)


def seed_library(library: Path) -> None:
    library.mkdir(parents=True, exist_ok=True)
    root_images = 8
    for index in range(root_images):
        make_photo(library / f"Photo-{index + 1:02d}.jpg", 1200, 800, index, noisy=False)
    fixtures = repo_root / "assets" / "face" / "fixtures"
    people = library / "People"
    people.mkdir(parents=True, exist_ok=True)
    if fixtures.is_dir():
        for fixture in sorted(fixtures.glob("*.png")):
            shutil.copy(fixture, people / fixture.name)
    else:
        print(f"warning: face fixtures not found at {fixtures}", file=sys.stderr)
    (library / "Empty Album" / "Day 1").mkdir(parents=True, exist_ok=True)


def seed_source(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    # Large, noisy files on purpose: the copy must stay in flight for a few
    # seconds so the tour can capture the import progress footer mid-run.
    for index in range(12):
        make_photo(source / f"Scan-{index + 1:03d}.jpg", 3200, 2134, 100 + index, noisy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    seed_library(args.library)
    seed_source(args.source)
    print(f"seeded {args.library}, {args.source}, {args.destination}")


if __name__ == "__main__":
    main()
