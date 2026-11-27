"""Render a `FleetSim` rollout plus a mission `Overlay` to mp4.

    film = FleetFilm(sim, "demo/x.mp4", size=(1280, 720), field=(xmin, xmax, ymin, ymax), camera="wide")
    film.frame(overlay)   # after each sim.step, or every k-th: the frame rate follows the tick gap
    n = film.close()

    ~/projects/newton-sandbox/.venv/bin/python newton_track/film_fleet.py     # demo, no mission layer

What is drawn, and with which viewer call:

    field boundary       log_lines   thin dark rectangle on the ground
    hazards              log_mesh    flat orange ring of the given radius (world-space width)
    beacon               log_instances + log_lines   dull red dome while buried; once found a bright
                                     red ball on a pole with two ground rings
    targets              log_instances + log_lines   small ground ball in the robot's role colour
                                     and a pale line from the robot's feet to it
    links                log_lines   violet line between the two robots' head markers
    roles                log_instances   ball above each head, colour from ROLE_COLORS
    caption, legend      burned into a band along the bottom with Pillow (already in the
                         Newton environment). Without Pillow the captions go to a
                         sidecar .srt next to the mp4 and the legend is dropped.

Every batch is rebuilt from the overlay on every frame, so anything that leaves
the overlay leaves the picture.

Same display caveats as film.py: `ViewerGL(headless=True)` needs a real screen,
and a Retina screen returns frames at twice the requested size, so the encoder
is opened from the first frame's actual shape. A clip is an illustration of one
seed, never a measurement.
"""

from __future__ import annotations

import math
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import warp as wp

import newton
import newton.viewer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fleet_sim import FleetSim, IcePatch, Overlay  # noqa: E402
from g1_newton import POLICY_HZ  # noqa: E402

try:  # Pillow ships with the Newton environment; captions degrade to .srt without it
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None

ROLE_COLORS = {
    "sweep": (0.10, 0.72, 0.25),  # green
    "locate": (1.00, 0.82, 0.00),  # yellow
    "converge": (0.92, 0.12, 0.78),  # magenta
    "hold": (0.12, 0.25, 0.95),  # blue
    "backoff": (0.00, 0.80, 0.82),  # cyan
    "down": (0.06, 0.06, 0.07),  # black
}
UNKNOWN_ROLE_COLOR = (0.55, 0.55, 0.55)
BEACON_BURIED = (0.55, 0.10, 0.08)
BEACON_FOUND = (1.00, 0.08, 0.05)
HAZARD_COLOR = (1.00, 0.33, 0.00)
LINK_COLOR = (0.50, 0.15, 1.00)
BOUNDARY_COLOR = (0.12, 0.16, 0.24)

FOV_DEG = 45.0  # the viewer camera's vertical field of view
# Camera orientation. Robots start on the ymin edge and walk +y, so a camera past
# the (xmax, ymax) corner looking back sees them three-quarter front, moving left
# to right and slightly toward the lens, with the 14 m axis along the wide side.
WIDE_YAW, WIDE_PITCH = -152.0, -30.0
FOLLOW_YAW, FOLLOW_PITCH = -152.0, -20.0
FOLLOW_PAD = 0.6  # m of ground kept around each upright robot
FOLLOW_MIN_DIST = 6.5  # m, so that robots standing together do not fill the frame
FOLLOW_MARGIN = (0.10, 0.10, 0.08, 0.30)
FOLLOW_SMOOTH = 0.08  # per-frame low-pass on the follow camera position
# Fractions of the half-frame kept clear: left, right, top, bottom. The bottom
# 68 of 720 lines carry the caption and the legend.
FRAME_MARGIN = (0.04, 0.04, 0.05, 0.21)
BAND_RGBA = (10, 14, 22, 150)

ROBOT_TOP = 1.55  # m, head marker height above the ground for a standing G1
HEAD_ABOVE_BASE = 0.74  # m from the pelvis to the head marker, standing
DOWN_ABOVE_BASE = 0.45  # m from the pelvis to the marker once fallen
GROUND_Z = 0.03  # overlays sit this far above the snow (ice patches top out at 2 mm)
RING_SEGMENTS = 48


def fit_camera(points, yaw_deg, pitch_deg, aspect, margin=FRAME_MARGIN, fov_deg=FOV_DEG) -> np.ndarray:
    """Closest camera position, for a fixed orientation, that keeps `points` in frame.

    In the camera's (right, up, front) basis each point gives four linear
    constraints on the position, so the closest feasible camera is a tiny LP
    that separates into a horizontal and a vertical 1-D problem with a closed
    form each.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    front = np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    up = np.cross(right, front)
    pts = np.asarray(points, dtype=float)
    pr, pu, pf = pts @ right, pts @ up, pts @ front
    tv = math.tan(math.radians(fov_deg) / 2)
    th = tv * aspect

    def axis(c, k_pos, k_neg):
        a = np.min(pf - c / k_pos)  # c_f <= a + c0 / k_pos   (points on the + side)
        b = np.min(pf + c / k_neg)  # c_f <= b - c0 / k_neg   (points on the - side)
        c0 = (b - a) / (1 / k_pos + 1 / k_neg)
        return a, b, a + c0 / k_pos

    ah, bh, fh = axis(pr, th * (1 - margin[1]), th * (1 - margin[0]))
    av, bv, fv = axis(pu, tv * (1 - margin[2]), tv * (1 - margin[3]))
    cf = min(fh, fv)
    # centre the slack axis inside its feasible interval
    cr = 0.5 * ((cf - ah) * th * (1 - margin[1]) + (bh - cf) * th * (1 - margin[0]))
    cu = 0.5 * ((cf - av) * tv * (1 - margin[2]) + (bv - cf) * tv * (1 - margin[3]))
    return cr * right + cu * up + cf * front


def _box_points(xmin, xmax, ymin, ymax, top=ROBOT_TOP):
    return [(x, y, z) for x in (xmin, xmax) for y in (ymin, ymax) for z in (0.0, top)]


def _circle(cx, cy, r, z, n=RING_SEGMENTS):
    a = np.linspace(0.0, 2 * math.pi, n + 1)
    pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n + 1, z)], axis=1)
    return pts[:-1], pts[1:]


def _font(px: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


class FleetFilm:
    """Fixed or following camera over a `FleetSim`, with the mission overlay drawn in."""

    def __init__(
        self,
        sim: FleetSim,
        out_path,
        size: tuple[int, int] = (1280, 720),
        field: tuple[float, float, float, float] | None = None,
        camera: str = "wide",
        *,
        fps: float | None = None,  # None: real time, from the tick gap between the first two frames
        legend: bool = True,
    ):
        if camera not in ("wide", "follow"):
            raise ValueError(f"camera must be 'wide' or 'follow', got {camera!r}")
        self.ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if not pathlib.Path(self.ffmpeg).exists():
            raise RuntimeError("ffmpeg not found")
        self.sim = sim
        self.out = pathlib.Path(out_path)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = int(size[0]), int(size[1])
        self.camera = camera
        self.fps = None if fps is None else float(fps)
        self._first = None  # (frame bytes, shape, sim.t) held until the frame rate is known
        self.legend = legend
        self.frames = 0
        self.burn = Image is not None
        self._enc = None
        self._closed = False
        self._cam_pos = None
        self._captions: list[tuple[int, str]] = []  # (first frame, text), for the .srt fallback
        self._strips: dict = {}
        self._hazard_key = None
        self._hazard_mesh = None
        self._balls_logged = False

        if field is None:  # no rectangle given: frame the starting line-up generously
            xy = self._base()[:, :2]
            field = (xy[:, 0].min() - 3, xy[:, 0].max() + 3, xy[:, 1].min() - 3, xy[:, 1].max() + 3)
        self.field = tuple(float(v) for v in field)

        self.viewer = v = newton.viewer.ViewerGL(width=self.w, height=self.h, headless=True)
        v.set_model(sim.model)
        r = v.renderer
        r.sky_upper = (0.30, 0.52, 0.86)
        r.sky_lower = (0.80, 0.88, 0.96)
        r.exposure = 0.85 * float(getattr(r, "exposure", 1.0))
        # The shadow map is a square of this half-width around the ORIGIN (default
        # 10 m); a field that reaches past it loses its shadows at the far end.
        x0, x1, y0, y1 = self.field
        r.shadow_extents = max(10.0, max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1)) + 2.0)
        # The default near plane of 1 cm leaves about 2 mm of depth resolution at
        # 20 m, the same as the ice patches' 2 mm step above the snow.
        v.camera.near = 0.2
        # Line width is in framebuffer pixels, and the framebuffer is 2x on Retina.
        r.line_width = 1.6 * v.renderer.window.get_framebuffer_size()[0] / self.w
        v.log_geo("/fleet/ball_mesh", newton.GeoType.SPHERE, (1.0,), 0.0, True, hidden=True)
        self.wide_pos = fit_camera(_box_points(*self.field), WIDE_YAW, WIDE_PITCH, self.w / self.h)

    # -- scene queries -----------------------------------------------------

    def _base(self) -> np.ndarray:
        """(n, 3) pelvis positions straight from the state being rendered."""
        return self.sim.state_0.joint_q.numpy().reshape(self.sim.n, self.sim.nq)[:, :3].astype(float)

    def _aim(self, base: np.ndarray):
        if self.camera == "wide":
            return self.wide_pos, WIDE_PITCH, WIDE_YAW
        up = base[~np.asarray(self.sim.down, dtype=bool)]
        if len(up):
            lo, hi = up[:, :2].min(axis=0) - FOLLOW_PAD, up[:, :2].max(axis=0) + FOLLOW_PAD
            box = _box_points(lo[0], hi[0], lo[1], hi[1], top=ROBOT_TOP + 0.15)
            want = fit_camera(box, FOLLOW_YAW, FOLLOW_PITCH, self.w / self.h, FOLLOW_MARGIN)
            y, p = math.radians(FOLLOW_YAW), math.radians(FOLLOW_PITCH)
            front = np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])
            centre = np.array([*up[:, :2].mean(axis=0), 0.8])
            want = want - front * max(0.0, FOLLOW_MIN_DIST - float((centre - want) @ front))
            if self._cam_pos is None:
                self._cam_pos = want
            else:
                self._cam_pos = self._cam_pos + FOLLOW_SMOOTH * (want - self._cam_pos)
        elif self._cam_pos is None:  # everyone is down on the first frame
            self._cam_pos = self.wide_pos
        return self._cam_pos, FOLLOW_PITCH, FOLLOW_YAW

    # -- overlay -----------------------------------------------------------

    def _draw_overlay(self, ov: Overlay, base: np.ndarray):
        v, n = self.viewer, self.sim.n
        down = np.asarray(self.sim.down, dtype=bool)
        k = 1.0 if self.camera == "wide" else 0.7  # markers shrink a little for the closer camera
        head = base.copy()
        # A fallen robot's pelvis can end up at or just under z = 0 (only its feet
        # and torso collide with the ground), so its marker is measured from the snow.
        head[:, 2] = np.maximum(base[:, 2], 0.0) + np.where(down, DOWN_ABOVE_BASE, HEAD_ABOVE_BASE)
        roles = list(ov.roles or [])
        role_col = [ROLE_COLORS.get(roles[r], UNKNOWN_ROLE_COLOR) if r < len(roles) else None for r in range(n)]

        starts, ends, lcols = [], [], []
        pts, radii, pcols = [], [], []

        def line(a, b, c):
            starts.append(a), ends.append(b), lcols.append(c)

        def loop(a, b, c):
            starts.extend(a), ends.extend(b), lcols.extend([c] * len(a))

        def ball(p, r, c):
            pts.append(p), radii.append(r), pcols.append(c)

        fld = ov.field if ov.field is not None else self.field
        if fld is not None:
            x0, x1, y0, y1 = fld
            c = [(x0, y0, GROUND_Z), (x1, y0, GROUND_Z), (x1, y1, GROUND_Z), (x0, y1, GROUND_Z)]
            for i in range(4):
                line(c[i], c[(i + 1) % 4], BOUNDARY_COLOR)

        if ov.beacon is not None:
            bx, by = ov.beacon
            if ov.beacon_found:
                top = (bx, by, 1.9)
                line((bx, by, 0.0), top, BEACON_FOUND)
                ball(top, 0.20 * k, BEACON_FOUND)
                for rr in (0.45, 0.80):
                    loop(*_circle(bx, by, rr, GROUND_Z), BEACON_FOUND)
            else:
                ball((bx, by, 0.02), 0.13 * k, BEACON_BURIED)  # half buried
                loop(*_circle(bx, by, 0.30, GROUND_Z), BEACON_BURIED)

        for r, tgt in enumerate(list(ov.targets or [])[:n]):
            if tgt is None:
                continue
            c = role_col[r] or UNKNOWN_ROLE_COLOR
            pale = tuple(0.45 * ch + 0.55 * 0.62 for ch in c)  # role tint washed toward grey
            line((base[r, 0], base[r, 1], GROUND_Z), (tgt[0], tgt[1], GROUND_Z), pale)
            ball((tgt[0], tgt[1], 0.03), 0.09 * k, c)

        for i, j in ov.links or []:
            if 0 <= i < n and 0 <= j < n and i != j:
                line(tuple(head[i]), tuple(head[j]), LINK_COLOR)

        for r in range(n):
            if role_col[r] is not None:
                ball(tuple(head[r]), 0.13 * k, role_col[r])

        dev = v.device
        if starts:
            v.log_lines(
                "/fleet/lines",
                wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3, device=dev),
                wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3, device=dev),
                wp.array(np.asarray(lcols, dtype=np.float32), dtype=wp.vec3, device=dev),
            )
        else:
            v.log_lines("/fleet/lines", None, None, None)
        # Balls are instances of one smooth sphere. `log_points` would do, but its
        # prototype is a 6 x 6 sphere that reads as a hexagon at this size.
        if pts:
            xf = np.zeros((len(pts), 7), dtype=np.float32)
            xf[:, :3], xf[:, 6] = pts, 1.0
            v.log_instances(
                "/fleet/balls",
                "/fleet/ball_mesh",
                wp.array(xf, dtype=wp.transform, device=dev),
                wp.array(np.repeat(np.asarray(radii, dtype=np.float32)[:, None], 3, axis=1), dtype=wp.vec3, device=dev),
                wp.array(np.asarray(pcols, dtype=np.float32), dtype=wp.vec3, device=dev),
                wp.array(np.tile(np.float32([[0.9, 0.0, 0.0, 0.0]]), (len(pts), 1)), dtype=wp.vec4, device=dev),
            )
            self._balls_logged = True
        elif self._balls_logged:
            v.log_instances("/fleet/balls", "/fleet/ball_mesh", None, None, None, None, hidden=True)
        self._draw_hazards(ov.hazards or [])

    def _draw_hazards(self, hazards):
        """All hazard rings as one flat mesh, rebuilt only when the list changes."""
        key = tuple((round(float(h[0]), 3), round(float(h[1]), 3), round(float(h[2]), 3)) for h in hazards)
        if key != self._hazard_key and key:
            half = 0.05 if self.camera == "wide" else 0.035  # half the ring's width on the ground, m
            a = np.linspace(0.0, 2 * math.pi, RING_SEGMENTS, endpoint=False)
            ca, sa = np.cos(a), np.sin(a)
            verts, tris = [], []
            for x, y, r in key:
                o = len(verts) * RING_SEGMENTS
                for rr in (max(r - half, 0.0), r + half):  # inner circle, then outer
                    verts.append(np.stack([x + rr * ca, y + rr * sa, np.full_like(a, GROUND_Z)], axis=1))
                for s in range(RING_SEGMENTS):
                    t = (s + 1) % RING_SEGMENTS
                    i0, i1, o0, o1 = o + s, o + t, o + RING_SEGMENTS + s, o + RING_SEGMENTS + t
                    tris += [i0, o0, o1, i0, o1, i1]  # counter-clockwise from above
            verts = np.concatenate(verts).astype(np.float32)
            dev = self.viewer.device
            self._hazard_mesh = (
                wp.array(verts, dtype=wp.vec3, device=dev),
                wp.array(np.asarray(tris, dtype=np.int32), dtype=wp.int32, device=dev),
                wp.array(np.tile(np.float32([[0.0, 0.0, 1.0]]), (len(verts), 1)), dtype=wp.vec3, device=dev),
            )
        self._hazard_key = key
        if self._hazard_mesh is None:
            return  # no hazard has ever been drawn
        p, i, nrm = self._hazard_mesh
        self.viewer.log_mesh(
            "/fleet/hazards", p, i, nrm, hidden=not key, backface_culling=False, color=HAZARD_COLOR, roughness=1.0,
            metallic=0.0, dynamic=True,
        )  # fmt: skip

    # -- 2-D burn-in -------------------------------------------------------

    def _caption_strip(self, text: str, fw: int, u: float):
        key = ("cap", text, fw)
        if key not in self._strips:
            px = round(26 * u)
            font = _font(px)
            while px > round(15 * u) and font.getlength(text) > fw - 40 * u:
                px -= max(1, round(u))
                font = _font(px)
            band = round(42 * u)
            img = Image.new("RGBA", (fw, band), BAND_RGBA)
            ImageDraw.Draw(img).text((fw / 2, band / 2), text, font=font, fill=(255, 255, 255, 255), anchor="mm")
            self._strips = {k: val for k, val in self._strips.items() if k[0] != "cap"}  # keep one caption cached
            self._strips[key] = np.asarray(img)
        return self._strips[key]

    def _legend_strip(self, fw: int, u: float):
        """One row along the bottom edge: the six roles, then beacon, hazard, link."""
        key = ("legend", fw)
        if key not in self._strips:
            font = _font(round(14 * u))
            items = [(name, col, "dot") for name, col in ROLE_COLORS.items()]
            items += [("beacon", BEACON_FOUND, "dot"), ("hazard", HAZARD_COLOR, "ring"), ("link", LINK_COLOR, "bar")]
            rad, gap, sep = round(6 * u), round(7 * u), round(24 * u)
            widths = [2 * rad + gap + font.getlength(name) for name, _, _ in items]
            band = round(26 * u)
            img = Image.new("RGBA", (fw, band), BAND_RGBA)
            d = ImageDraw.Draw(img)
            x, cy = (fw - sum(widths) - sep * (len(items) - 1)) / 2, band // 2
            for (name, col, kind), wd in zip(items, widths):
                rgb = tuple(round(255 * ch) for ch in col) + (255,)
                box = (x, cy - rad, x + 2 * rad, cy + rad)
                if kind == "dot":
                    d.ellipse(box, fill=rgb, outline=(235, 235, 235, 255), width=max(1, round(u)))
                elif kind == "ring":
                    d.ellipse(box, outline=rgb, width=max(2, round(2 * u)))
                else:
                    d.line((x, cy, x + 2 * rad, cy), fill=rgb, width=max(2, round(2 * u)))
                d.text((x + 2 * rad + gap, cy), name, font=font, fill=(255, 255, 255, 255), anchor="lm")
                x += wd + sep
            self._strips[key] = np.asarray(img)
        return self._strips[key]

    @staticmethod
    def _blend(frame: np.ndarray, strip: np.ndarray, x: int, y: int):
        sh, sw = strip.shape[:2]
        sh, sw = min(sh, frame.shape[0] - y), min(sw, frame.shape[1] - x)
        a = strip[:sh, :sw, 3:4].astype(np.uint16)
        dst = frame[y : y + sh, x : x + sw].astype(np.uint16)
        frame[y : y + sh, x : x + sw] = ((strip[:sh, :sw, :3] * a + dst * (255 - a)) // 255).astype(np.uint8)

    # -- public ------------------------------------------------------------

    def render(self, overlay: Overlay | None = None) -> np.ndarray:
        """Draw `sim.state_0` plus `overlay` and return the RGB frame at capture resolution."""
        ov = overlay if overlay is not None else Overlay()
        v = self.viewer
        base = self._base()
        pos, pitch, yaw = self._aim(base)
        v.set_camera(wp.vec3(*(float(c) for c in pos)), pitch=pitch, yaw=yaw)
        v.begin_frame(self.sim.t / POLICY_HZ)
        v.log_state(self.sim.state_0)
        self._draw_overlay(ov, base)
        v.end_frame()
        frame = v.get_frame().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame[..., :3])
        if self.burn:
            fh, fw = frame.shape[:2]
            u = fh / 720  # text is sized for 720 lines, and the capture is 2x on a Retina screen
            y = fh
            if self.legend:
                strip = self._legend_strip(fw, u)
                y -= strip.shape[0]
                self._blend(frame, strip, 0, y)
            if ov.caption:
                strip = self._caption_strip(ov.caption, fw, u)
                self._blend(frame, strip, 0, y - strip.shape[0])
        return frame

    def frame(self, overlay: Overlay | None = None):
        if self._closed:
            raise RuntimeError("FleetFilm is closed")
        frame = self.render(overlay)
        if self._enc is None and self.fps is None and self._first is None:
            # Filming every k-th tick must still play in real time, and ffmpeg wants
            # the rate up front: hold the first frame until the second gives the gap.
            self._first = (frame.tobytes(), frame.shape, self.sim.t)
        else:
            if self._enc is None:
                if self.fps is None:
                    self.fps = POLICY_HZ / max(1, self.sim.t - self._first[2])
                self._open(frame.shape)
            self._enc.stdin.write(frame.tobytes())
        caption = (overlay.caption if overlay is not None else "") or ""
        if not self._captions or self._captions[-1][1] != caption:
            self._captions.append((self.frames, caption))
        self.frames += 1

    def _open(self, shape):
        fh, fw = shape[:2]  # the capture's real size, not `size`: 2x on a Retina screen
        self._enc = subprocess.Popen(
            [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw}x{fh}",
             "-r", f"{self.fps:g}", "-i", "-", "-vf", f"scale={self.w}:{self.h}", "-pix_fmt", "yuv420p",
             "-crf", "20", str(self.out)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        if self._first is not None:
            self._enc.stdin.write(self._first[0])
            self._first = None

    def close(self) -> int:
        """Finish the mp4 and return the number of frames written."""
        if self._closed:
            return self.frames
        self._closed = True
        rc = 0
        if self._enc is None and self._first is not None:  # a one-frame film
            self.fps = float(POLICY_HZ)
            self._open(self._first[1])
        if self._enc is not None:
            self._enc.stdin.close()
            rc = self._enc.wait()
        self.viewer.close()
        if not self.burn and any(text for _, text in self._captions):
            self._write_srt()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with {rc} writing {self.out}")
        return self.frames

    def _write_srt(self):
        def stamp(f):
            ms = round(1000 * f / self.fps)
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

        ends = [*self._captions[1:], (self.frames, "")]
        cues = [(a, b, text) for (a, text), (b, _) in zip(self._captions, ends) if text]
        body = "".join(f"{k}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n" for k, (a, b, text) in enumerate(cues, 1))
        self.out.with_suffix(".srt").write_text(body, encoding="utf-8")


# -- demo: no mission layer, a scripted overlay that exercises every field -----


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def demo(out: str = "demo/newton_fleet_demo.mp4", camera: str = "wide", ticks: int = 1150, every: int = 2) -> int:
    field = (-6.0, 6.0, 0.0, 14.0)
    lanes = (-4.5, -1.5, 1.5, 4.5)
    n = len(lanes)
    beacon = (-3.0, 9.5)
    sim = FleetSim(
        [(x, 0.5, math.pi / 2) for x in lanes],
        [IcePatch(1.5, 5.0, 1.1, 1.4)],  # in robot 2's lane only
    )
    film = FleetFilm(sim, out, field=field, camera=camera)

    roles = ["sweep"] * n
    targets: list = [(x, field[3] - 0.5) for x in lanes]
    hazards: list = []
    found = False
    caption = "Sweep: 4 robots in line, 3 m apart"
    fell_at = found_at = None
    flash: dict = {}  # link -> last tick it is shown
    for t in range(ticks):
        # scripted "mission": just enough to drive every overlay field
        for r in range(n):
            if sim.down[r] and roles[r] != "down":
                x, y, _ = sim.pose(r)
                roles[r], targets[r], fell_at = "down", None, t
                hazards.append((x, y, 1.6))
                caption = f"Robot {r} down on ice: hazard mapped, neighbours told"
                flash.update({(r, j): t + 40 for j in range(n) if j != r})
                if r + 1 < n and not sim.down[r + 1]:
                    roles[r + 1], targets[r + 1] = "backoff", None
        if fell_at is not None and t == fell_at + 120:
            roles = ["hold" if role == "backoff" else role for role in roles]
        if not found and t == ticks // 2 - 100 and not sim.down[0]:
            roles[0] = "locate"
            caption = "Robot 0 has a signal: locating"
        if not found and t == ticks // 2:
            found, found_at = True, t
            caption = "Beacon found: robots 0 and 1 converge"
            flash.update({(0, j): t + 40 for j in range(1, n)})
            for r in (0, 1):
                if not sim.down[r]:
                    roles[r], targets[r] = "converge", beacon
        if t % 50 == 0:  # routine neighbour heartbeat
            alive = [r for r in range(n) if not sim.down[r]]
            flash.update({(a, b): t + 8 for a, b in zip(alive, alive[1:])})

        cmd = np.zeros((n, 3), dtype=np.float32)
        for r in range(n):
            x, y, yaw = sim.pose(r)
            want, speed = math.pi / 2, 0.5
            if roles[r] == "converge":
                dx, dy = beacon[0] - x, beacon[1] - y
                if math.hypot(dx, dy) < 0.9:
                    roles[r], targets[r] = "hold", None
                else:
                    want = math.atan2(dy, dx)
            if roles[r] == "backoff":
                speed = -0.3
            elif roles[r] in ("hold", "down"):
                speed = 0.0
            elif y > field[3] - 0.8:
                roles[r], targets[r], speed = "hold", None, 0.0
            cmd[r] = (speed, 0.0, float(np.clip(1.5 * _wrap(want - yaw), -1.0, 1.0)))
        sim.step(cmd)

        if t % every == 0:
            film.frame(
                Overlay(
                    field=field, beacon=beacon, beacon_found=found, hazards=list(hazards),
                    links=[pair for pair, until in flash.items() if t <= until], targets=list(targets),
                    roles=list(roles), caption=caption,
                )
            )  # fmt: skip
    print(f"down: {sim.down.tolist()}, fell at tick {fell_at}, beacon found at tick {found_at}, final roles {roles}")
    frames = film.close()
    print(f"wrote {out} ({frames} frames, camera {camera})")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fleet film demo with a scripted overlay (no mission layer).")
    ap.add_argument("--out", default="demo/newton_fleet_demo.mp4")
    ap.add_argument("--camera", default="wide", choices=("wide", "follow"))
    ap.add_argument("--ticks", type=int, default=1150)
    ap.add_argument("--every", type=int, default=2, help="film every k-th 20 ms tick")
    args = ap.parse_args()
    sys.exit(demo(args.out, args.camera, args.ticks, args.every))
