#!/usr/bin/env python3
"""
ScreenPeek — shoulder-surfing detector for macOS.

Watches the built-in camera. When it sees more than one face at the same time
(you + someone peeking over your shoulder), it saves a timestamped snapshot
and opens Photo Booth so the snooper gets a nice portrait of themselves.

Usage:
    python3 screenpeek.py                 # start watching
    python3 screenpeek.py --check         # self-test, launches nothing
    python3 screenpeek.py --preview       # live window with boxes, for tuning
    python3 screenpeek.py --min-faces 3   # only trigger on 3+ people

Ctrl+C to stop.

Detection
---------
Default backend is YuNet, a small neural-net face detector that ships with
OpenCV >= 4.5.4. It is ~5x faster than the classic Haar cascade and, unlike
Haar, copes with turned heads, glasses and poor light — which is exactly the
"someone leaning in from the side" case. Haar is kept as a fallback
(--backend haar) for very old OpenCV builds.

Frames are pulled continuously on a background thread so the detector always
sees the *latest* image, not a stale one from the camera's buffer. A trigger
needs the extra face to persist for --confirm seconds (default 0.8s) in most
of the samples in that window, so a single flicker never fires it, but a real
peek fires in under a second.
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit(
        "OpenCV is missing. Install it with:\n"
        "    python3 -m pip install --user opencv-python\n"
        "(or run ./install.sh)"
    )


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = Path.home() / "Pictures" / "ScreenPeek"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"
HAAR_NAME = "haarcascade_frontalface_default.xml"
LOG_FILE = APP_DIR / "screenpeek.log"


def say(msg):
    """Print with a timestamp and append the same line to screenpeek.log."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as fh:
            fh.write(datetime.now().strftime("%Y-%m-%d ") + line + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# detectors
#   Each returns a list of (x, y, w, h, score) in full-frame pixel coords.
# --------------------------------------------------------------------------

class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path, min_score):
        self.min_score = min_score
        self._size = None
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            score_threshold=min_score, nms_threshold=0.3, top_k=50,
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(frame)
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, fw, fh = (int(v) for v in row[:4])
            score = float(row[14])
            if score >= self.min_score:
                out.append((max(0, x), max(0, y), fw, fh, score))
        return out


class HaarDetector:
    name = "haar"

    def __init__(self, model_path, sensitivity):
        self.sensitivity = sensitivity
        self._cc = cv2.CascadeClassifier(str(model_path))
        if self._cc.empty():
            raise RuntimeError(f"OpenCV could not parse {model_path}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(30, int(gray.shape[0] * 0.08))
        faces = self._cc.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=self.sensitivity,
            minSize=(min_side, min_side), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in faces]


def find_model(name):
    """Look next to the script first, then inside the OpenCV package."""
    candidates = [APP_DIR / name]
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.append(Path(data_dir) / name)
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def load_detector(backend, min_score, sensitivity, quiet=False):
    """Build the requested detector, falling back Haar if YuNet can't load."""
    want_yunet = backend in ("auto", "yunet")
    if want_yunet:
        model = find_model(YUNET_NAME)
        if model is None:
            reason = f"{YUNET_NAME} not found next to screenpeek.py"
        elif not hasattr(cv2, "FaceDetectorYN"):
            reason = f"OpenCV {cv2.__version__} is too old for YuNet (need >= 4.5.4)"
        else:
            try:
                return YuNetDetector(model, min_score)
            except cv2.error as exc:
                reason = f"YuNet failed to load: {str(exc).strip().splitlines()[-1]}"
        if backend == "yunet":
            sys.exit(reason)
        if not quiet:
            print(f"  ! {reason}; falling back to the Haar cascade.")

    model = find_model(HAAR_NAME)
    if model is None:
        sys.exit(
            f"Could not find {HAAR_NAME} (or {YUNET_NAME}) next to screenpeek.py "
            f"in {APP_DIR}.\nRe-download the app folder."
        )
    try:
        return HaarDetector(model, sensitivity)
    except RuntimeError as exc:
        sys.exit(f"{exc} — the file is likely truncated. Re-download the app folder.")


def filter_faces(faces, frame_shape, min_face_frac):
    """Drop boxes too small to be a person in the room (posters, phone screens)."""
    min_h = frame_shape[0] * min_face_frac
    return [f for f in faces if f[3] >= min_h]


# --------------------------------------------------------------------------
# camera — background grabber so we always analyse the freshest frame
# --------------------------------------------------------------------------

class Camera:
    def __init__(self, index, width):
        self.index = index
        self.width = width
        self._cap = None
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.fps = 0.0

    def start(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)  # try the default backend
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.width * 9 / 16))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Let auto-exposure settle before the first real read.
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        self._cap = cap
        with self._lock:
            self._frame = frame
            self._seq = 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        n, t0 = 0, time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
            n += 1
            if n >= 30:
                now = time.monotonic()
                self.fps = n / (now - t0)
                n, t0 = 0, now

    def latest(self):
        """Return (frame, seq). seq lets callers skip a frame they've seen."""
        with self._lock:
            return self._frame, self._seq

    def alive(self):
        return self._thread is not None and self._thread.is_alive() and self._cap.isOpened()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._thread = None


def open_camera(index, width, attempts=3, wait=2.0):
    for i in range(attempts):
        cam = Camera(index, width)
        if cam.start():
            return cam
        if i < attempts - 1:
            time.sleep(wait)
    return None


def reacquire_camera(args):
    """Get the camera back after handing it to another app. Never gives up:
    another app may hold it for a while, and quitting is the wrong answer."""
    cam = open_camera(args.camera, args.width, attempts=3, wait=1.0)
    if cam is not None:
        return cam
    say("  ! camera still busy (Photo Booth / Preview has it); retrying every 2s…")
    n = 0
    while cam is None:
        time.sleep(2)
        n += 1
        cam = open_camera(args.camera, args.width, attempts=1)
        if cam is None and n % 15 == 0:      # a reminder every ~30s
            say(f"  ! still waiting for the camera ({n * 2}s). "
                "Close anything using it (Zoom, FaceTime, Photo Booth).")
    say("  ✓ camera is back")
    return cam


# --------------------------------------------------------------------------
# reaction
# --------------------------------------------------------------------------

def draw_faces(img, faces, color):
    for (x, y, w, h, score) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if score < 1.0:
            cv2.putText(img, f"{score:.2f}", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_snapshot(frame, faces, outdir, annotate=True):
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = outdir / f"peek_{stamp}_{len(faces)}faces.jpg"
    # Filenames are second-resolution; never overwrite evidence.
    dupe = 2
    while path.exists():
        path = outdir / f"peek_{stamp}_{len(faces)}faces_{dupe}.jpg"
        dupe += 1
    shot = frame.copy()
    if annotate:
        draw_faces(shot, faces, (0, 0, 255))
        label = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"  ({len(faces)} faces)"
        cv2.putText(shot, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), shot, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return path


def log_event(outdir, faces, snapshot):
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = outdir / "peeks.csv"
    new = not logfile.exists()
    with logfile.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["timestamp", "faces", "snapshot"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         len(faces), snapshot.name])


def open_photo_booth():
    try:
        subprocess.run(["open", "-a", "Photo Booth"], check=True,
                       capture_output=True, timeout=15)
        return True
    except FileNotFoundError:
        print("  ! 'open' not found — is this actually macOS?")
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode(errors="replace").strip()
        print(f"  ! Could not launch Photo Booth: {msg or 'unknown error'}")
    except subprocess.TimeoutExpired:
        print("  ! Photo Booth took too long to launch.")
    return False


def photo_booth_running():
    try:
        return subprocess.run(["pgrep", "-x", "Photo Booth"],
                              capture_output=True, timeout=5).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def close_photo_booth():
    """Quit Photo Booth. Polite AppleScript first, firm pkill if that fails.

    The AppleScript route may show a one-time macOS prompt asking whether
    Terminal may control Photo Booth; either answer is fine because pkill
    is the fallback and needs no permission.
    """
    if not photo_booth_running():
        return True
    try:
        subprocess.run(["osascript", "-e", 'tell application "Photo Booth" to quit'],
                       capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    for _ in range(10):                      # give it up to ~1s to go away
        if not photo_booth_running():
            return True
        time.sleep(0.1)
    try:
        subprocess.run(["pkill", "-x", "Photo Booth"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    for _ in range(10):
        if not photo_booth_running():
            return True
        time.sleep(0.1)
    return False


def open_file(path):
    """Open a file in its default app (Preview for a JPEG)."""
    try:
        subprocess.run(["open", str(path)], check=True,
                       capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return False


def beep():
    subprocess.Popen(
        ["afplay", "/System/Library/Sounds/Submarine.aiff"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# --------------------------------------------------------------------------
# decision: "has the extra face been there for real?"
# --------------------------------------------------------------------------

class Confirmer:
    """Sliding time window of yes/no samples.

    Fires when the window spans at least `seconds`, holds at least
    `min_samples`, and at least `ratio` of the samples were positive.
    Tolerates the odd missed detection; ignores a one-frame flicker.
    """

    def __init__(self, seconds, ratio, min_samples=3):
        self.seconds = seconds
        self.ratio = ratio
        self.min_samples = min_samples
        self._buf = deque()

    def add(self, positive, now=None):
        now = time.monotonic() if now is None else now
        self._buf.append((now, bool(positive)))
        cutoff = now - self.seconds
        # Drop stale samples, but always keep one at or before the cutoff:
        # it anchors the window so the measured span can actually reach
        # `seconds`. (Trimming it too would leave the span forever a bit
        # short whenever samples arrive faster than the window length.)
        while len(self._buf) > 2 and self._buf[1][0] < cutoff:
            self._buf.popleft()
        return self.confirmed()

    def confirmed(self):
        if len(self._buf) < self.min_samples:
            return False
        span = self._buf[-1][0] - self._buf[0][0]
        if span < self.seconds - 1e-6:  # tolerate float rounding
            return False
        positives = sum(1 for _, p in self._buf if p)
        return positives / len(self._buf) >= self.ratio

    def fraction(self):
        if not self._buf:
            return 0.0
        return sum(1 for _, p in self._buf if p) / len(self._buf)

    def reset(self):
        self._buf.clear()


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def self_check(args):
    """Prove the whole pipeline works, without opening Photo Booth."""
    print("ScreenPeek self-check\n")
    print(f"  python      {sys.version.split()[0]}")
    print(f"  opencv      {cv2.__version__}")

    detector = load_detector(args.backend, args.min_score, args.sensitivity)
    model = find_model(YUNET_NAME if detector.name == "yunet" else HAAR_NAME)
    print(f"  detector    {detector.name}  ({model})")

    outdir = Path(args.dir).expanduser()
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        probe = outdir / ".screenpeek_write_test"
        probe.write_text("ok")
        probe.unlink()
        print(f"  snapshots   {outdir} (writable)")
    except OSError as exc:
        sys.exit(f"  snapshots   cannot write to {outdir}: {exc}")

    print(f"\n  opening camera {args.camera}...")
    cam = open_camera(args.camera, args.width, attempts=1)
    if cam is None:
        sys.exit(
            "  FAILED. macOS needs to grant camera access to Terminal:\n"
            "    System Settings > Privacy & Security > Camera > Terminal\n"
            "  Then quit Terminal completely (Cmd+Q) and reopen it.\n"
            "  Also close anything else using the camera (Zoom, FaceTime, Photo Booth)."
        )
    frame, _ = cam.latest()
    print(f"  got a {frame.shape[1]}x{frame.shape[0]} frame\n")

    print(f"  Counting faces for {args.check_seconds:.0f} seconds.")
    print("  Sit in front of the camera, then have someone lean in next to you.\n")

    deadline = time.monotonic() + args.check_seconds
    seen, ms_samples, last_seq = [], [], 0
    faces = []
    try:
        while time.monotonic() < deadline:
            frame, seq = cam.latest()
            if seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            t = time.perf_counter()
            faces = filter_faces(detector.detect(frame), frame.shape, args.min_face)
            ms_samples.append((time.perf_counter() - t) * 1000)
            seen.append(len(faces))
            scores = " ".join(f"{f[4]:.2f}" for f in faces) if detector.name == "yunet" else ""
            print(f"    {len(faces)} face(s)  {'#' * len(faces) or '-'}  {scores}")
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\n  stopped early")

    snapshot = save_snapshot(frame, faces, outdir)
    cam.stop()

    best = max(seen) if seen else 0
    avg_ms = sum(ms_samples) / len(ms_samples) if ms_samples else 0
    print(f"\n  camera         {cam.fps:.0f} fps")
    print(f"  detection      {avg_ms:.1f} ms per frame ({detector.name})")
    print(f"  most faces     {best} at once")
    print(f"  test snapshot  {snapshot}")

    if best >= args.min_faces:
        print(f"\n  Working. It will trigger at {args.min_faces}+ faces held for "
              f"{args.confirm:.1f}s.")
    elif best >= 1:
        print("\n  It sees you, but never saw a second person. If someone WAS")
        print("  leaning in, retry with:  ./run.sh --check --min-score 0.6")
    else:
        print("\n  No faces detected at all. Try better lighting or face the camera,")
        print("  then retry with:  ./run.sh --check --min-score 0.6")
    print("\n  Nothing was launched — this was a test. Start for real with ./run.sh")


def watch(args):
    detector = load_detector(args.backend, args.min_score, args.sensitivity)
    outdir = Path(args.dir).expanduser()

    cam = open_camera(args.camera, args.width)
    if cam is None:
        sys.exit(
            f"Could not open camera {args.camera}.\n"
            "On macOS the app running this script needs camera permission:\n"
            "  System Settings → Privacy & Security → Camera → enable Terminal.\n"
            "Also make sure nothing else (Zoom, Photo Booth, FaceTime) is hogging it."
        )

    confirmer = Confirmer(args.confirm, args.confirm_ratio)
    say(f"ScreenPeek watching ({detector.name}). Trigger at {args.min_faces}+ faces "
        f"held for {args.confirm:.1f}s, checking every {args.interval:.2f}s.")
    say(f"Re-arms when the extra face leaves for {args.rearm:g}s "
        f"(or after {args.cooldown:g}s at most).")
    say(f"Snapshots → {outdir}")
    say(f"Log → {LOG_FILE}")
    print("Ctrl+C to stop.\n")

    armed = True
    last_trigger = float("-inf")   # monotonic() counts from boot; 0 would be wrong
    clear_since = None             # when the scene last dropped below min_faces
    last_seq = 0
    triggers = 0

    try:
        while True:
            loop_start = time.monotonic()

            if not cam.alive():
                say("  . lost the camera, retrying")
                cam.stop()
                time.sleep(3)
                cam = reacquire_camera(args)
                confirmer.reset()
                last_seq = 0
                continue

            frame, seq = cam.latest()
            if seq == last_seq:
                # No new frame since last check — don't re-count a stale image.
                time.sleep(0.01)
                continue
            last_seq = seq

            faces = filter_faces(detector.detect(frame), frame.shape, args.min_face)
            positive = len(faces) >= args.min_faces

            if args.preview:
                shown = frame.copy()
                draw_faces(shown, faces, (0, 255, 0))
                cv2.putText(shown,
                            f"{len(faces)} face(s)  confirm {confirmer.fraction():.0%}  "
                            f"{cam.fps:.0f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("ScreenPeek preview (q to quit)", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.monotonic()

            if not armed:
                # Re-arm once the extra face has been gone for --rearm seconds,
                # or after --cooldown seconds regardless (so a peeker who never
                # leaves still gets a second portrait eventually).
                if positive:
                    clear_since = None
                else:
                    clear_since = clear_since if clear_since is not None else now
                gone = clear_since is not None and (now - clear_since) >= args.rearm
                if gone or (now - last_trigger) >= args.cooldown:
                    armed = True
                    confirmer.reset()
                    say("  ✓ armed again" + (" (extra face left)" if gone else " (cooldown over)"))

            elif confirmer.add(positive, now):
                triggers += 1
                say(f"{len(faces)} faces — someone's peeking. (trigger #{triggers})")

                snapshot = save_snapshot(frame, faces, outdir, annotate=not args.clean)
                log_event(outdir, faces, snapshot)
                say(f"  → saved {snapshot.name}")

                if args.sound:
                    beep()

                if not args.no_photo_booth:
                    # Hand the camera over: Photo Booth wants it too, and on
                    # some Macs it will show a black frame otherwise.
                    cam.stop()
                    if open_photo_booth():
                        say("  → Photo Booth is up. Say cheese.")
                        if args.show_for > 0:
                            time.sleep(args.show_for)
                            if close_photo_booth():
                                say(f"  → Photo Booth closed after {args.show_for:g}s.")
                            else:
                                say("  ! Could not close Photo Booth; leaving it open.")
                    else:
                        say("  ! Photo Booth did not launch.")

                if not args.no_open_snapshot:
                    say("  → opened the snapshot." if open_file(snapshot)
                        else "  ! Could not open the snapshot.")

                if not args.no_photo_booth:
                    # Grace period for Photo Booth to release the camera, then
                    # keep trying until we get it back. Never give up here.
                    time.sleep(args.handover)
                    cam = reacquire_camera(args)
                    last_seq = 0

                armed = False
                clear_since = None
                last_trigger = time.monotonic()
                say(f"  … watching for the extra face to leave "
                    f"(re-arms after {args.rearm:g}s clear, or {args.cooldown:g}s)")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if cam is not None:
            cam.stop()
        if args.preview:
            cv2.destroyAllWindows()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Open Photo Booth (and save a snapshot) when more than one "
                    "person is looking at your Mac.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("what counts as a peek")
    g.add_argument("--min-faces", type=int, default=2,
                   help="how many faces count as 'someone is peeking'")
    g.add_argument("--confirm", type=float, default=0.8,
                   help="seconds the extra face must persist before triggering")
    g.add_argument("--confirm-ratio", type=float, default=0.7,
                   help="fraction of checks in that window that must see it "
                        "(tolerates the odd missed detection)")
    g.add_argument("--rearm", type=float, default=1.0,
                   help="after a trigger, re-arm once the extra face has been "
                        "gone for this many seconds")
    g.add_argument("--cooldown", type=float, default=20.0,
                   help="…or after this many seconds regardless, even if the "
                        "peeker never leaves")

    g = p.add_argument_group("detection")
    g.add_argument("--backend", choices=["auto", "yunet", "haar"], default="auto",
                   help="face detector; auto = YuNet if available, else Haar")
    g.add_argument("--min-score", type=float, default=0.75,
                   help="YuNet confidence needed to count a face (0.5-0.95). "
                        "Lower catches more, false-alarms more")
    g.add_argument("--min-face", type=float, default=0.06,
                   help="ignore faces shorter than this fraction of frame height "
                        "(filters posters, phone screens, people far away)")
    g.add_argument("--sensitivity", type=int, default=6,
                   help="Haar only: minNeighbors, 4-8 is the useful range")
    g.add_argument("--interval", type=float, default=0.2,
                   help="seconds between checks")

    g = p.add_argument_group("camera and output")
    g.add_argument("--dir", default=str(DEFAULT_DIR),
                   help="where snapshots and peeks.csv go")
    g.add_argument("--camera", type=int, default=0, help="camera index")
    g.add_argument("--width", type=int, default=640,
                   help="capture width; 640 is a good balance, 1280 sees "
                        "further but costs more CPU")
    g.add_argument("--show-for", type=float, default=3.0,
                   help="seconds Photo Booth stays open before ScreenPeek closes "
                        "it again; 0 = leave it open")
    g.add_argument("--handover", type=float, default=1.5,
                   help="seconds to wait for Photo Booth to release the camera "
                        "before taking it back")
    g.add_argument("--clean", action="store_true",
                   help="save snapshots without boxes and timestamp")
    g.add_argument("--no-open-snapshot", action="store_true",
                   help="don't open the saved snapshot in Preview after a trigger")
    g.add_argument("--sound", action="store_true",
                   help="also play an alert sound on trigger")
    g.add_argument("--no-photo-booth", action="store_true",
                   help="only save snapshots, never launch Photo Booth")

    g = p.add_argument_group("modes")
    g.add_argument("--preview", action="store_true",
                   help="show a live window with detection boxes, for tuning")
    g.add_argument("--check", action="store_true",
                   help="self-test: verify model, camera and folder, print a "
                        "live face count and timings, then exit. Launches nothing.")
    g.add_argument("--check-seconds", type=float, default=15.0,
                   help="how long --check watches for")
    return p.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    if _args.check:
        self_check(_args)
    else:
        watch(_args)
