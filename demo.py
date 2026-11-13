"""Interactive 2D affine transform playground.
Install: pip install numpy opencv-python
Run: python affine_transform_playground.py [optional_image_path]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Affine Transform Playground"
W, H = 900, 600
ORIGIN = (W // 2, H // 2)

# Each slider is an integer. Scale/shear values are divided by 100.
SLIDERS = {
    "a x-scale": (-200, 200, 100),
    "b x-shear": (-200, 200, 0),
    "c x-move": (-250, 250, 0),
    "d y-shear": (-200, 200, 0),
    "e y-scale": (-200, 200, 100),
    "f y-move": (-250, 250, 0),
}
PRESETS = {
    ord("0"): (1, 0, 0, 0, 1, 0),
    ord("1"): (2, 0, 0, 0, 2, 0),
    ord("2"): (.5, 0, 0, 0, .5, 0),
    ord("3"): (1, .7, 0, 0, 1, 0),
    ord("4"): (1, 0, 0, .7, 1, 0),
    ord("5"): (1, 0, 90, 0, 1, 60),
    ord("6"): (0, 1, 0, -1, 0, 0),
    ord("7"): (0, -1, 0, 1, 0, 0),
    ord("8"): (-1, 0, 0, 0, 1, 0),
    ord("9"): (1, 0, 0, 0, -1, 0),
}

def noop(_):
    pass

def values_to_sliders(values):
    for (name, (minimum, maximum, _)), value in zip(SLIDERS.items(), values):
        integer = round(value * 100) if name[0] in "abde" else round(value)
        cv2.setTrackbarPos(name, WINDOW, integer - minimum)

def get_values():
    values = []
    for name, (minimum, _, _) in SLIDERS.items():
        raw = cv2.getTrackbarPos(name, WINDOW) + minimum
        values.append(raw / 100 if name[0] in "abde" else raw)
    return values

def make_source():
    """Image with asymmetry makes rotations and reflections obvious."""
    img = np.full((H, W, 3), (245, 245, 245), dtype=np.uint8)
    cv2.rectangle(img, (ORIGIN[0], ORIGIN[1]),
                  (ORIGIN[0] + 120, ORIGIN[1] + 90), (235, 160, 70), -1)
    cv2.rectangle(img, (ORIGIN[0], ORIGIN[1]),
                  (ORIGIN[0] + 120, ORIGIN[1] + 90), (135, 75, 20), 2)
    cv2.circle(img, (ORIGIN[0] + 25, ORIGIN[1] + 20), 9, (30, 30, 220), -1)
    cv2.putText(img, "A", (ORIGIN[0] + 65, ORIGIN[1] + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (25, 25, 25), 3, cv2.LINE_AA)
    return img

def load_source():
    if len(sys.argv) < 2:
        return make_source()
    path = Path(sys.argv[1])
    src = cv2.imread(str(path))
    if src is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    # Keep the chosen image centered in a 900x600 canvas.
    scale = min(400 / src.shape[1], 350 / src.shape[0], 1.0)
    width, height = max(1, int(src.shape[1] * scale)), max(1, int(src.shape[0] * scale))
    resized = cv2.resize(src, (width, height))
    canvas = np.full((H, W, 3), 245, dtype=np.uint8)
    x, y = ORIGIN[0] - width // 2, ORIGIN[1] - height // 2
    canvas[y:y+height, x:x+width] = resized
    return canvas

def draw_axes(canvas):
    ox, oy = ORIGIN
    cv2.line(canvas, (0, oy), (W-1, oy), (150,150,150), 1)
    cv2.line(canvas, (ox, 0), (ox, H-1), (150,150,150), 1)
    cv2.putText(canvas, "x", (W-25, oy-12), cv2.FONT_HERSHEY_SIMPLEX, .7, (70,70,70), 2)
    cv2.putText(canvas, "y", (ox+10, H-16), cv2.FONT_HERSHEY_SIMPLEX, .7, (70,70,70), 2)

def main():
    src = load_source()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1020, 950)
    for name, (lo, hi, default) in SLIDERS.items():
        cv2.createTrackbar(name, WINDOW, default-lo, hi-lo, noop)
    print("Keys: 0 reset | 1 zoom 2x | 2 zoom 0.5x | 3 horizontal shear | 4 vertical shear")
    print("      5 translate | 6 CCW 90 | 7 CW 90 | 8 mirror X | 9 mirror Y | Q/Esc exit")
    while True:
        a, b, c, d, e, f = get_values()
        # In image coordinates, +y points DOWN. Origin-centered affine transform:
        # [x_new, y_new] = A @ ([x, y] - origin) + origin + [c, f]
        matrix = np.float32([
            [a, b, ORIGIN[0]*(1-a) - b*ORIGIN[1] + c],
            [d, e, ORIGIN[1]*(1-e) - d*ORIGIN[0] + f],
        ])
        transformed = cv2.warpAffine(src, matrix, (W, H),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(245,245,245))
        draw_axes(transformed)
        det = a*e-b*d
        cv2.rectangle(transformed, (6, 6), (W-7, 92), (245,245,245), -1)
        cv2.putText(transformed, f"[{a:5.2f} {b:5.2f} {c:5.0f}]   x' = {a:.2f} x + {b:.2f} y + {c:.0f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .57, (20,20,20), 1, cv2.LINE_AA)
        cv2.putText(transformed, f"[{d:5.2f} {e:5.2f} {f:5.0f}]   y' = {d:.2f} x + {e:.2f} y + {f:.0f}",
                    (15, 57), cv2.FONT_HERSHEY_SIMPLEX, .57, (20,20,20), 1, cv2.LINE_AA)
        cv2.putText(transformed, f"det={det:.2f}  | keys 0-9 presets, q quit",
                    (15, 82), cv2.FONT_HERSHEY_SIMPLEX, .53, (70,70,70), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, transformed)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in PRESETS:
            values_to_sliders(PRESETS[key])
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
