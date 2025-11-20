#!/usr/bin/env python3
"""Generate a side-by-side comparison between PES thread colors and a render."""

import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from pyembroidery import EmbPattern


def resolve_path(base_dir: str, candidate: str, default_folder: str, *, must_exist: bool = True) -> str:
    """Return an absolute path for the provided CLI argument."""
    if os.path.isabs(candidate):
        return os.path.abspath(candidate)

    preferred = os.path.join(base_dir, candidate)
    if os.path.exists(preferred):
        return os.path.abspath(preferred)
    if not must_exist:
        return os.path.abspath(preferred)

    fallback = os.path.join(base_dir, default_folder, candidate)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)

    raise FileNotFoundError(f"Could not resolve {candidate} via {preferred} or {fallback}")


def load_thread_palette(pes_path: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Read the PES file and return (label, rgb) tuples for each thread."""
    pattern = EmbPattern()
    pattern.read(pes_path)

    palette = []
    for idx, thread in enumerate(pattern.threadlist, start=1):
        name = thread.description or thread.catalog_number or f"Thread {idx}"
        rgb = (thread.get_red(), thread.get_green(), thread.get_blue())
        palette.append((f"#{idx:02d} {name}", rgb))

    if not palette:
        raise RuntimeError(f"No thread colors found in {pes_path}")

    return palette


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font loading with graceful fallback."""
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_color(rgb: Tuple[int, int, int]) -> str:
    """Choose white or black text depending on luminance for readability."""
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 150 else "#ffffff"


def build_palette_panel(
    colors: List[Tuple[str, Tuple[int, int, int]]],
    width: int,
    target_height: int,
) -> Image.Image:
    """Render labeled color blocks stacked vertically."""
    block_height = 48
    min_spacing = 8
    required_height = len(colors) * block_height

    if len(colors) > 1:
        required_height += (len(colors) + 1) * min_spacing
    else:
        required_height += min_spacing * 2

    canvas_height = max(required_height, target_height)
    available_spacing = canvas_height - len(colors) * block_height
    spacing = max(min_spacing, available_spacing // (len(colors) + 1))

    panel = Image.new("RGB", (width, canvas_height), color="#ffffff")
    draw = ImageDraw.Draw(panel)
    font = _load_font(18)

    y = spacing
    padding = 16
    for label, rgb in colors:
        top_left = (padding, y)
        bottom_right = (width - padding, y + block_height)
        draw.rectangle([top_left, bottom_right], fill=rgb)

        text = f"{label}  RGB{rgb}"
        text_color = _text_color(rgb)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = y + (block_height - text_height) / 2
        draw.text((padding + 12, text_y), text, fill=text_color, font=font)

        y += block_height + spacing

    title = "Original Thread Palette"
    title_font = _load_font(22)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_bbox[2] - title_bbox[0])) / 2
    draw.text((title_x, max(4, spacing // 3)), title, fill="#111111", font=title_font)

    return panel


def build_comparison_image(
    palette_image: Image.Image,
    render_image: Image.Image,
) -> Image.Image:
    """Place the palette on the left and the render on the right."""
    render_rgb = render_image.convert("RGB")
    total_height = max(palette_image.height, render_rgb.height)
    total_width = palette_image.width + render_rgb.width

    composite = Image.new("RGB", (total_width, total_height), color="#ffffff")

    palette_y = (total_height - palette_image.height) // 2
    render_y = (total_height - render_rgb.height) // 2
    composite.paste(palette_image, (0, palette_y))
    composite.paste(render_rgb, (palette_image.width, render_y))

    return composite


def ensure_folder(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a color-palette comparison image for a PES render."
    )
    parser.add_argument(
        "--pes",
        default="input_PES/cat.PES",
        help="Path to the PES file or name relative to input_PES/.",
    )
    parser.add_argument(
        "--render",
        default="output/cat_fast_cli.png",
        help="Rendered PNG to place on the right side.",
    )
    parser.add_argument(
        "--output",
        default="output/cat_palette_comparison.png",
        help="Destination PNG for the comparison image.",
    )
    parser.add_argument(
        "--palette_width",
        type=int,
        default=420,
        help="Width in pixels for the color palette column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    pes_path = resolve_path(repo_root, args.pes, "input_PES")
    render_path = resolve_path(repo_root, args.render, "output")
    output_path = resolve_path(repo_root, args.output, "output", must_exist=False)

    print(f"Loading PES colors from {pes_path}")
    colors = load_thread_palette(pes_path)

    print(f"Opening render image from {render_path}")
    render_image = Image.open(render_path)

    print("Building palette panel...")
    palette_image = build_palette_panel(colors, width=args.palette_width, target_height=render_image.height)

    print("Generating composite comparison...")
    comparison = build_comparison_image(palette_image, render_image)

    ensure_folder(output_path)
    comparison.save(output_path)
    print(f"Saved comparison image to {output_path}")


if __name__ == "__main__":
    main()
