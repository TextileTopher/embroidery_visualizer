#!/usr/bin/env python3
"""Composite a render on top of a repeating fabric texture."""

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image


def resolve_path(base_dir: str, candidate: str, fallback_folder: str, *, must_exist: bool = True) -> str:
    if os.path.isabs(candidate):
        path = os.path.abspath(candidate)
        if not must_exist or os.path.exists(path):
            return path
        raise FileNotFoundError(f"{path} does not exist")

    direct = os.path.join(base_dir, candidate)
    if os.path.exists(direct):
        return os.path.abspath(direct)

    fallback = os.path.join(base_dir, fallback_folder, candidate)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)

    if not must_exist:
        return os.path.abspath(direct)

    raise FileNotFoundError(f"Could not resolve {candidate} (checked {direct} and {fallback})")


def ensure_folder(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def tile_texture(texture: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = size
    tiled = Image.new("RGB", size)
    tex_w, tex_h = texture.size

    for y in range(0, height, tex_h):
        for x in range(0, width, tex_w):
            tiled.paste(texture, (x, y))

    return tiled


def replace_white_with_texture(render: Image.Image, texture: Image.Image, tolerance: int) -> Image.Image:
    """Swap near-white pixels with the provided texture."""
    render_rgba = render.convert("RGBA")
    tiled_texture = tile_texture(texture, render_rgba.size).convert("RGBA")

    render_np = np.array(render_rgba, dtype=np.uint8)
    texture_np = np.array(tiled_texture, dtype=np.uint8)

    rgb = render_np[..., :3].astype(np.int16)
    diff = 255 - rgb
    mask = (np.abs(diff) <= tolerance).all(axis=-1)

    render_np[mask, :3] = texture_np[mask, :3]
    render_np[mask, 3] = 255

    return Image.fromarray(render_np, mode="RGBA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay an embroidery render onto a fabric texture.")
    parser.add_argument("--render", default="output/cat_fast_cli.png", help="PNG/JPG render to composite.")
    parser.add_argument(
        "--texture",
        default="assets/textures/canvas1.png",
        help="Texture image that will sit underneath the render.",
    )
    parser.add_argument(
        "--output",
        default="output/cat_canvas_overlay.png",
        help="Destination path for the composited image.",
    )
    parser.add_argument(
        "--white_tolerance",
        type=int,
        default=8,
        help="How close a pixel must be to pure white before it is replaced (default: 8).",
    )
    return parser.parse_args()


def apply_texture_overlay(
    render_path: str,
    texture_path: str,
    output_path: str,
    tolerance: int = 8,
) -> None:
    render = Image.open(render_path)
    texture = Image.open(texture_path).convert("RGB")

    composite = replace_white_with_texture(render, texture, tolerance)
    ensure_folder(output_path)
    composite.save(output_path)


def main() -> None:
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    render_path = resolve_path(repo_root, args.render, "output")
    texture_path = resolve_path(repo_root, args.texture, "assets/textures")
    output_path = resolve_path(repo_root, args.output, "output", must_exist=False)

    apply_texture_overlay(
        render_path=render_path,
        texture_path=texture_path,
        output_path=output_path,
        tolerance=args.white_tolerance,
    )
    print(f"Saved textured composite to {output_path}")


if __name__ == "__main__":
    main()
