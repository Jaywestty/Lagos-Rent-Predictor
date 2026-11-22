"""Draw the Adugbo AI app icons (the mascot's head) as PNGs for the installable web app.

Run:  uv run --with pillow python scripts/make_app_icons.py
Writes frontend/public/icons/: icon-192.png, icon-512.png (transparent corners),
maskable-512.png (full-bleed background with a safe zone) and apple-touch-icon.png (180, opaque).
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "icons"

SHELL = "#f1ebde"
SHELL_EDGE = "#d8ccb6"
SCREEN = "#0c3328"
GLOW = "#8ff0c6"
ROOF = "#0f4a37"
PIN = "#f39a2b"
GROUND = "#f6f7f5"

# The head lives in the SVG box x 28..212, y -26..164 (see frontend/src/components/brand/Mascot.tsx).
BOX_X, BOX_Y, BOX_W, BOX_H = 28, -26, 184, 190


def _quad(p0, p1, p2, steps=160):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / steps for i in range(steps + 1))]


def draw_head(size: int, fill: float, background: str | None, radius_pct: float = 0.0) -> Image.Image:
    """Render at 4x then downsample for smooth edges. `fill` is how much of the canvas the head spans."""
    big = size * 4
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if background:
        d.rounded_rectangle([0, 0, big - 1, big - 1], radius=int(big * radius_pct), fill=background)

    scale = big * fill / max(BOX_W, BOX_H)
    ox = (big - BOX_W * scale) / 2 - BOX_X * scale
    oy = (big - BOX_H * scale) / 2 - BOX_Y * scale

    def P(x, y):
        return (ox + x * scale, oy + y * scale)

    def rect(x, y, w, h, r, color, outline=None, width=0):
        d.rounded_rectangle([P(x, y), P(x + w, y + h)], radius=r * scale, fill=color, outline=outline, width=int(width * scale))

    def stroke(points, width, color):
        # A smooth round-capped stroke: overlapping discs along the curve (no ragged joins).
        r = width * scale / 2
        for x, y in (P(*pt) for pt in points):
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    rect(50, 56, 140, 106, 36, SHELL, SHELL_EDGE, 3)
    rect(66, 76, 108, 70, 26, SCREEN)
    stroke(_quad((88, 108), (99, 94), (110, 108)), 9, GLOW)
    stroke(_quad((130, 108), (141, 94), (152, 108)), 9, GLOW)
    stroke(_quad((108, 124), (120, 135), (132, 124)), 8, GLOW)
    d.polygon([P(34, 78), P(120, 18), P(206, 78), P(194, 90), P(120, 38), P(46, 90)], fill=ROOF)
    # map pin: a circle with a point underneath
    d.polygon([P(103, 4), P(137, 4), P(120, 30)], fill=PIN)
    cx, cy = P(120, -4)
    r = 20 * scale
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PIN)
    r2 = 7.5 * scale
    d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill="#ffffff")

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_head(192, 0.86, GROUND, 0.22).save(OUT / "icon-192.png")
    draw_head(512, 0.86, GROUND, 0.22).save(OUT / "icon-512.png")
    # Maskable: the OS crops to a circle/squircle, so keep the head inside the central 70%.
    draw_head(512, 0.62, GROUND).save(OUT / "maskable-512.png")
    draw_head(180, 0.80, GROUND).convert("RGB").save(OUT / "apple-touch-icon.png")
    print("Wrote icons to", OUT)


if __name__ == "__main__":
    main()
