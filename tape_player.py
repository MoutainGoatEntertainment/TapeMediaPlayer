#!/usr/bin/env python3
"""
tape_player.py — A cassette-tape-style audio player.

Install dependencies:
    uv run --with PySide6 --with sounddevice --with numpy tape_player.py
    brew install ffmpeg          # required to decode MP3 and WAV

Controls:
    Hold ◀◀  / Left arrow  : rewind at 4× (audible, backward)
    ▶ / ⏸   / Space        : play / pause
    Hold ▶▶  / Right arrow : fast-forward at 4× (audible, pitch-shifted)

Releasing ◀◀ or ▶▶ restores whatever play/pause state was active before
you pressed the button — exactly like a real tape deck.

Drag MP3/WAV files onto the cassette, or click Open.
All files are joined into one continuous tape in alphabetical order.
"""

import sys
import math
import threading

import av
import numpy as np
import sounddevice as sd
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QBrush, QFont, QFontMetrics,
    QRadialGradient, QLinearGradient, QPixmap, QIcon,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
CHANNELS    = 2
FF_SPEED    = 4.0
BLOCKSIZE   = 2048


# ── Audio engine ──────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Real-time tape engine.

    The entire tape (all loaded files concatenated) lives in self.tape as a
    float32 numpy array of shape (N, 2).  A playback cursor (self.pos) moves
    through the array at a rate of `speed` samples per output sample:

        speed =  1.0  → normal forward play   (pitch unchanged)
        speed =  4.0  → fast-forward           (pitch rises 4×, like a real tape)
        speed = -4.0  → rewind                 (backward + pitch rise)
    """

    def __init__(self):
        self.tape:    np.ndarray | None = None   # (N, 2) float32
        self.pos      = 0.0
        self.speed    = 1.0
        self.playing  = False
        self._lock    = threading.Lock()
        self._stream  = None

        self.track_boundaries: list[int] = []
        self.track_names:      list[str] = []
        self.track_artists:    list[str] = []
        self._prev_pos = 0.0
        self._saved_playing = False   # state preserved across FF/RW

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_files(self, paths: list[str]) -> bool:
        """Decode and concatenate audio files into a single tape array via PyAV."""
        segments, boundaries, names, artists, cur = [], [], [], [], 0
        for path in paths:
            try:
                arr = self._decode(path)
                boundaries.append(cur)
                names.append(Path(path).stem)
                artists.append(self._read_artist(path))
                cur += len(arr)
                segments.append(arr)
            except Exception as exc:
                print(f"Skipping {path}: {exc}", file=sys.stderr)

        if not segments:
            return False

        tape = np.concatenate(segments)
        with self._lock:
            self.tape             = tape
            self.pos              = 0.0
            self._prev_pos        = 0.0
            self.playing          = False
            self.speed            = 1.0
            self.track_boundaries = boundaries
            self.track_names      = names
            self.track_artists    = artists
        return True

    @staticmethod
    def _read_artist(path: str) -> str:
        """Return the artist tag from an audio file, or '' if absent."""
        try:
            with av.open(path) as container:
                meta = container.metadata
                return (meta.get("artist") or meta.get("TPE1") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _decode(path: str) -> np.ndarray:
        """Decode any audio file to a (N, 2) float32 array at SAMPLE_RATE using PyAV."""
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
        frames = []
        with av.open(path) as container:
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    frames.append(out.to_ndarray().T)   # (samples, channels)
            for out in resampler.resample(None):         # flush
                frames.append(out.to_ndarray().T)
        if not frames:
            raise ValueError("No audio decoded")
        return np.concatenate(frames).astype(np.float32)

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def shutdown(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── Transport ─────────────────────────────────────────────────────────────

    def toggle_play(self):
        with self._lock:
            if self.tape is None:
                return
            self.playing = not self.playing
            if self.playing:
                self.speed = 1.0

    def set_ff(self, active: bool):
        with self._lock:
            if self.tape is None:
                return
            if active:
                self._saved_playing = self.playing
                self.speed   = FF_SPEED
                self.playing = True
            else:
                self.speed   = 1.0
                self.playing = self._saved_playing

    def set_rw(self, active: bool):
        with self._lock:
            if self.tape is None:
                return
            if active:
                self._saved_playing = self.playing
                self.speed   = -FF_SPEED
                self.playing = True
            else:
                self.speed   = 1.0
                self.playing = self._saved_playing

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self.tape is not None

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self.playing

    @property
    def position(self) -> float:
        with self._lock:
            return self.pos

    @property
    def total(self) -> int:
        with self._lock:
            return len(self.tape) if self.tape is not None else 1

    def pop_delta(self) -> float:
        """Return signed samples consumed since last call (for reel animation)."""
        with self._lock:
            d = self.pos - self._prev_pos
            self._prev_pos = self.pos
        return d

    def _current_index(self) -> int:
        """Return index of the track at the current position (must hold lock)."""
        p = int(self.pos)
        idx = 0
        for i, b in enumerate(self.track_boundaries):
            if p >= b:
                idx = i
        return idx

    def current_track(self) -> str:
        with self._lock:
            if not self.track_names:
                return ""
            return self.track_names[self._current_index()]

    def current_artist(self) -> str:
        with self._lock:
            if not self.track_artists:
                return ""
            return self.track_artists[self._current_index()]

    def at_end(self) -> bool:
        with self._lock:
            if self.tape is None:
                return False
            return self.pos >= len(self.tape) - 1

    # ── Audio callback (called from sounddevice thread) ───────────────────────

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if self.tape is None or not self.playing:
                outdata[:] = 0
                return

            n     = len(self.tape)
            pos   = self.pos
            speed = self.speed

            # For each output frame i, read tape[pos + i*speed].
            # speed>0 → forward (pitch up if >1), speed<0 → backward.
            idx     = (pos + np.arange(frames, dtype=np.float64) * speed).astype(np.int64)
            new_pos = pos + frames * speed

            valid      = (idx >= 0) & (idx < n)
            out        = np.zeros((frames, CHANNELS), dtype=np.float32)
            out[valid] = self.tape[idx[valid]]
            outdata[:] = out

            self.pos = float(np.clip(new_pos, 0.0, float(n - 1)))

            if speed > 0 and self.pos >= n - 1:
                self.playing = False
            elif speed < 0 and self.pos <= 0.0:
                self.playing = False


# ── Cassette widget ───────────────────────────────────────────────────────────

class CassetteWidget(QWidget):
    """Cassette image with animated reel, label, and control overlays."""

    files_dropped  = Signal(list)
    open_requested = Signal()

    # ── Zone fractions — measured precisely from tape_markup.png (583×378) ────
    _LEFT_REEL_FX  = 0.298   # left reel centre x
    _RIGHT_REEL_FX = 0.703   # right reel centre x
    _REEL_FY       = 0.448   # both reel centres y
    _REEL_RFX      = 0.057   # reel radius (fraction of image width)

    _TITLE_FX = 0.115;  _TITLE_FY = 0.108   # "My Mixtape" zone
    _TITLE_FW = 0.768;  _TITLE_FH = 0.071

    _TRACK_FX = 0.070;  _TRACK_FY = 0.217   # current track name zone
    _TRACK_FW = 0.858;  _TRACK_FH = 0.058

    _PROG_FX  = 0.3945; _PROG_FY  = 0.3571  # tape progress window
    _PROG_FW  = 0.2127; _PROG_FH  = 0.1508

    _CTRL_FX  = 0.163;  _CTRL_FY  = 0.585   # controls strip (yellow)
    _CTRL_FW  = 0.672;  _CTRL_FH  = 0.122

    def __init__(self, engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.engine      = engine
        self.left_angle  = 0.0
        self.right_angle = 0.0
        self.pixmap      = QPixmap(str(Path(__file__).parent / "tape.png"))
        self.setMinimumSize(500, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)

        # ── Overlay buttons (children, positioned in resizeEvent) ─────────────
        self.rw_btn   = QPushButton("◀◀",   self)
        self.play_btn = QPushButton("▶",    self)
        self.ff_btn   = QPushButton("▶▶",   self)
        self.open_btn = QPushButton("Open…", self)

        for btn in (self.rw_btn, self.play_btn, self.ff_btn):
            btn.setStyleSheet(TRANSPORT_STYLE)
        self.open_btn.setStyleSheet(OPEN_STYLE)

        self.rw_btn.pressed.connect(lambda: engine.set_rw(True))
        self.rw_btn.released.connect(lambda: engine.set_rw(False))
        self.ff_btn.pressed.connect(lambda: engine.set_ff(True))
        self.ff_btn.released.connect(lambda: engine.set_ff(False))
        self.open_btn.clicked.connect(self.open_requested)

    # ── Drag & drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [
            u.toLocalFile()
            for u in event.mimeData().urls()
            if u.toLocalFile().lower().endswith((".mp3", ".wav"))
        ]
        if paths:
            self.files_dropped.emit(paths)

    # ── Layout ────────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        ir = self._img_rect()
        rw, rh = ir.width(), ir.height()
        rx, ry = ir.x(), ir.y()

        zx = rx + self._CTRL_FX * rw
        zy = ry + self._CTRL_FY * rh
        zw = self._CTRL_FW * rw
        zh = self._CTRL_FH * rh

        bw  = int(zw / 4)
        bh  = int(zh * 0.82)                  # slightly smaller than the zone
        byo = int((zh - bh) / 2)              # vertical offset to centre in zone
        for i, btn in enumerate((self.rw_btn, self.play_btn, self.ff_btn, self.open_btn)):
            btn.setGeometry(int(zx + i * bw), int(zy + byo), bw, bh)

    def _img_rect(self) -> QRectF:
        W, H = self.width(), self.height()
        iw, ih = self.pixmap.width(), self.pixmap.height()
        scale  = min(W / iw, H / ih)
        rw, rh = iw * scale, ih * scale
        return QRectF((W - rw) / 2, (H - rh) / 2, rw, rh)

    # ── Reel animation ────────────────────────────────────────────────────────

    def advance(self, delta: float):
        """Rotate both reels CCW by the amount of tape that just moved."""
        if not self.engine.is_loaded or delta == 0:
            return
        # Both always spin counter-clockwise (angle decreases = CCW in screen coords)
        arc = abs(delta) / SAMPLE_RATE * 180.0
        self.left_angle  -= arc
        self.right_angle -= arc
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw(p)

    def _draw(self, p: QPainter):
        ir = self._img_rect()
        rw, rh = ir.width(), ir.height()
        rx, ry = ir.x(), ir.y()

        # ── Cassette image ────────────────────────────────────────────────────
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(ir, self.pixmap, QRectF(self.pixmap.rect()))

        pos   = self.engine.position
        total = self.engine.total
        frac  = (pos / total) if total > 0 else 0.0

        # ── Animated reels ────────────────────────────────────────────────────
        max_r    = self._REEL_RFX * rw
        reel_y   = ry + self._REEL_FY * rh
        left_cx  = rx + self._LEFT_REEL_FX  * rw
        right_cx = rx + self._RIGHT_REEL_FX * rw
        self._draw_reel(p, left_cx,  reel_y, max_r, 1.0 - frac, self.left_angle)
        self._draw_reel(p, right_cx, reel_y, max_r, frac,        self.right_angle)

        # ── Tape progress window (purple zone) ────────────────────────────────
        if self.engine.is_loaded:
            px = rx + self._PROG_FX * rw;  py = ry + self._PROG_FY * rh
            pw = self._PROG_FW * rw;       ph = self._PROG_FH * rh
            self._draw_progress(p, px, py, pw, ph, frac, left_cx, right_cx, reel_y, rw)

        # ── Artist / title (blue zone) ────────────────────────────────────────
        tx = rx + self._TITLE_FX * rw;  ty = ry + self._TITLE_FY * rh
        tw = self._TITLE_FW * rw;       th = self._TITLE_FH * rh
        title_txt = self.engine.current_artist() or "My Mixtape"
        self._draw_label(p, tx, ty, tw, th, title_txt, size_frac=0.82)

        # ── Current track name (green zone) ──────────────────────────────────
        kx = rx + self._TRACK_FX * rw;  ky = ry + self._TRACK_FY * rh
        kw = self._TRACK_FW * rw;       kh = self._TRACK_FH * rh
        if self.engine.at_end():
            track_txt = "— end of tape —"
        elif self.engine.is_loaded:
            track_txt = self.engine.current_track()
        else:
            track_txt = "drop files here"
        self._draw_label(p, kx, ky, kw, kh, track_txt, size_frac=0.82)

    _HANDWRITTEN_FONT: str = ""

    @staticmethod
    def _pick_handwritten_font() -> str:
        if CassetteWidget._HANDWRITTEN_FONT:
            return CassetteWidget._HANDWRITTEN_FONT
        from PySide6.QtGui import QFontDatabase
        available = set(QFontDatabase.families())
        for name in ("Bradley Hand", "Noteworthy", "Marker Felt", "Comic Sans MS"):
            if name in available:
                CassetteWidget._HANDWRITTEN_FONT = name
                return name
        CassetteWidget._HANDWRITTEN_FONT = "Helvetica Neue"
        return CassetteWidget._HANDWRITTEN_FONT

    def _draw_label(self, p, x, y, w, h, text, size_frac=0.45):
        f = QFont(self._pick_handwritten_font())
        f.setPixelSize(max(8, int(h * size_frac)))
        p.setFont(f)
        pad    = int(w * 0.02)
        elided = QFontMetrics(f).elidedText(text, Qt.ElideRight, int(w - pad * 2))
        p.setPen(QColor("#1a3a6b"))
        p.drawText(QRectF(x + pad, y, w - pad * 2, h), Qt.AlignCenter, elided)

    def _draw_progress(self, p, px, py, pw, ph, frac,
                       left_cx, right_cx, reel_cy, rw):
        """Tape window: large brown circles at real reel positions, clipped to window rect."""
        # Radii sized so full circle nearly fills the window at frac=0/1
        # (reel centres at 0.298 and 0.703 of rw; window spans 0.403–0.602)
        tape_max_r = rw * 0.20   # full spool — edge reaches well into window
        # Min radius must exceed the gap between reel centre and window edge
        # (left gap = 0.105*rw, right gap = 0.101*rw) so the hub still peeks in
        tape_min_r = rw * 0.115  # empty hub — just visible at window edge

        r_left  = tape_min_r + (tape_max_r - tape_min_r) * (1.0 - frac)
        r_right = tape_min_r + (tape_max_r - tape_min_r) * frac

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(px, py, pw, ph), 3, 3)
        p.save()
        p.setClipPath(clip)

        # Dark window background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(10, 8, 5, 220)))
        p.drawRect(QRectF(px, py, pw, ph))

        # Tape circles — colour sampled from tape_circles.png (#6c5353)
        p.setBrush(QBrush(QColor("#6c5353")))
        p.drawEllipse(QPointF(left_cx,  reel_cy), r_left,  r_left)
        p.drawEllipse(QPointF(right_cx, reel_cy), r_right, r_right)

        p.restore()

    def _draw_reel(self, p: QPainter,
                   cx: float, cy: float, max_r: float,
                   fill: float, angle: float):
        # fill is unused — both reels always look identical
        hub_r  = max_r * 0.72
        axle_r = max_r * 0.14

        p.setPen(Qt.NoPen)

        # 1 — dark window background
        p.setBrush(QBrush(QColor("#111111")))
        p.drawEllipse(QPointF(cx, cy), max_r, max_r)

        # 2 — hub disk (dark gray, matching tape.png sample ~#3a3a38)
        p.setBrush(QBrush(QColor("#3a3a38")))
        p.drawEllipse(QPointF(cx, cy), hub_r, hub_r)

        # 3 — five dark wedge-gaps between the spokes
        gap_deg = 24   # angular width of each gap
        p.setBrush(QBrush(QColor("#111111")))
        for i in range(5):
            a_start = angle + i * 72 - gap_deg / 2
            # Qt uses 1/16th-degree units, CCW from 3 o'clock; negate to match screen CW/CCW
            p.drawPie(
                int(cx - hub_r), int(cy - hub_r),
                int(hub_r * 2),  int(hub_r * 2),
                int(-a_start * 16),
                int(-gap_deg  * 16),
            )

        # 4 — small centre ring (hub core)
        p.setBrush(QBrush(QColor("#484846")))
        p.drawEllipse(QPointF(cx, cy), hub_r * 0.28, hub_r * 0.28)

        # 5 — axle hole
        p.setBrush(QBrush(QColor("#111111")))
        p.drawEllipse(QPointF(cx, cy), axle_r, axle_r)

        # 6 — window rim
        p.setPen(QPen(QColor("#333333"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), max_r, max_r)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    _load_done = Signal(bool, int)

    def __init__(self):
        super().__init__()
        self.engine = AudioEngine()
        self.engine.start()

        self.setWindowTitle("Tape Player")
        self.setMinimumSize(520, 400)

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(16, 16, 16, 12)
        vbox.setSpacing(10)

        # Cassette (contains all controls)
        self.cassette = CassetteWidget(self.engine)
        self.cassette.files_dropped.connect(self._load_files)
        self.cassette.open_requested.connect(self._open_dialog)
        self.cassette.play_btn.clicked.connect(self._toggle_play)
        vbox.addWidget(self.cassette, stretch=1)

        # Status line
        self.status = QLabel("Drop MP3 / WAV files onto the cassette, or click Open")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #666; font-size: 11px;")
        vbox.addWidget(self.status)

        # Signals & timer
        self._load_done.connect(self._on_load_done)

        self._timer = QTimer()
        self._timer.setInterval(33)   # ≈30 fps
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.setStyleSheet(
            "QMainWindow, QWidget { background-color: #181818; color: #cccccc; }"
        )

    # ── Tick (animation + UI sync) ────────────────────────────────────────────

    def _tick(self):
        delta = self.engine.pop_delta()
        if delta:
            self.cassette.advance(delta)
        else:
            self.cassette.update()

        self.cassette.play_btn.setText("⏸" if self.engine.is_playing else "▶")

    # ── Transport ─────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if not self.engine.is_loaded:
            self._open_dialog()
            return
        self.engine.toggle_play()

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key_Right:
            self.engine.set_ff(True)
        elif key == Qt.Key_Left:
            self.engine.set_rw(True)
        elif key == Qt.Key_Space:
            self._toggle_play()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key_Right:
            self.engine.set_ff(False)
        elif key == Qt.Key_Left:
            self.engine.set_rw(False)

    # ── File loading ──────────────────────────────────────────────────────────

    def _open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Audio Files", "",
            "Audio Files (*.mp3 *.wav);;All Files (*)",
        )
        if paths:
            self._load_files(paths)

    def _load_files(self, paths: list):
        self.status.setText("Loading…")
        paths = sorted(paths)

        def _worker():
            ok = self.engine.load_files(paths)
            self._load_done.emit(ok, len(paths))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_done(self, ok: bool, count: int):
        if ok:
            n = count
            self.status.setText(
                f"Loaded {n} track{'s' if n != 1 else ''} as one tape · press ▶ or Space to play"
            )
        else:
            self.status.setText(
                "Could not load any files. Is ffmpeg installed? (brew install ffmpeg)"
            )
        self.cassette.left_angle  = 0.0
        self.cassette.right_angle = 0.0
        self.cassette.update()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.engine.shutdown()
        super().closeEvent(event)


# ── Styles ────────────────────────────────────────────────────────────────────

TRANSPORT_STYLE = """
QPushButton {
    background-color: #2e2e2e;
    color: #dddddd;
    border: 1px solid #4a4a4a;
    border-radius: 8px;
    font-size: 18px;
    padding: 2px 14px;
}
QPushButton:pressed { background-color: #525252; border-color: #707070; }
QPushButton:hover   { background-color: #383838; }
"""

OPEN_STYLE = """
QPushButton {
    background-color: #1d3a1d;
    color: #88cc88;
    border: 1px solid #2d5a2d;
    border-radius: 8px;
    font-size: 13px;
    padding: 2px 18px;
}
QPushButton:pressed { background-color: #2d5a2d; }
QPushButton:hover   { background-color: #254a25; }
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "tape.png")))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
