"""
ScreenPeek test suite. No camera or macOS needed — the camera and the
system commands (open / osascript / pkill) are replaced with fakes.

Run:   python3 tests/test_screenpeek.py
  or:  python3 -m pytest tests/
"""
import csv
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import screenpeek as pg  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TMP = Path(tempfile.mkdtemp(prefix="screenpeek-test-"))

ASTRO = cv2.imread(str(FIXTURES / "astronaut.png"))   # NASA, public domain
assert ASTRO is not None, "missing tests/fixtures/astronaut.png"
NOISE = (np.random.RandomState(0).rand(480, 640, 3) * 255).astype("uint8")
ONE = ASTRO
_a = cv2.resize(ASTRO, (320, 320))
TWO = np.hstack([_a, cv2.flip(_a, 1)])                 # two faces, 640x320


def out_dir(name):
    d = TMP / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


# ----------------------------------------------------------------- detectors

def test_backend_selection():
    assert pg.load_detector("auto", 0.75, 6, quiet=True).name == "yunet"
    assert pg.load_detector("haar", 0.75, 6, quiet=True).name == "haar"


def test_yunet_finds_faces():
    yu = pg.load_detector("yunet", 0.75, 6, quiet=True)
    f = yu.detect(ASTRO)
    assert len(f) == 1 and f[0][4] > 0.9
    assert yu.detect(NOISE) == []
    assert len(pg.filter_faces(yu.detect(TWO), TWO.shape, 0.06)) == 2


def test_yunet_handles_tilted_head():
    yu = pg.load_detector("yunet", 0.75, 6, quiet=True)
    M = cv2.getRotationMatrix2D((256, 256), 25, 1.0)
    tilted = cv2.warpAffine(ASTRO, M, (512, 512), borderValue=(90, 90, 90))
    assert len(yu.detect(tilted)) == 1


def test_haar_fallback_works():
    ha = pg.load_detector("haar", 0.75, 6, quiet=True)
    assert len(ha.detect(ASTRO)) >= 1


def test_min_face_filter():
    boxes = [(100, 100, 90, 88, 0.95), (600, 10, 20, 20, 0.9)]
    kept = pg.filter_faces(boxes, (480, 640), 0.06)      # 6% of 480 = 28.8 px
    assert kept == [boxes[0]]


def test_yunet_is_faster_than_haar():
    yu = pg.load_detector("yunet", 0.75, 6, quiet=True)
    ha = pg.load_detector("haar", 0.75, 6, quiet=True)
    t = time.perf_counter()
    for _ in range(20):
        yu.detect(NOISE)
    yu_ms = (time.perf_counter() - t) / 20
    t = time.perf_counter()
    for _ in range(5):
        ha.detect(NOISE)
    ha_ms = (time.perf_counter() - t) / 5
    assert yu_ms < ha_ms


# ----------------------------------------------------------------- confirmer

def test_confirmer_ignores_flicker():
    c = pg.Confirmer(0.8, 0.7)
    assert not any(c.add(i == 7, now=1000 + i * 0.2) for i in range(20))


def test_confirmer_fires_after_window():
    c = pg.Confirmer(0.8, 0.7)
    fired_at = next(i * 0.2 for i in range(10) if c.add(True, now=1000 + i * 0.2))
    assert 0.8 <= fired_at <= 1.0


def test_confirmer_ratio():
    c = pg.Confirmer(0.8, 0.7)
    r = [c.add(p, now=1000 + i * 0.2) for i, p in enumerate([1, 1, 0, 1, 1])]
    assert r[-1] is True                  # 4 of 5 = 80% -> fires
    c.reset()
    r = [c.add(p, now=1000 + i * 0.2) for i, p in enumerate([1, 0, 1, 0, 1])]
    assert r[-1] is False                 # 3 of 5 = 60% -> does not


def test_confirmer_with_jittered_timing():
    import random
    rnd = random.Random(1)
    for _ in range(200):
        c = pg.Confirmer(0.8, 0.7)
        t = first = None
        t = 0.0
        for _i in range(40):
            t += 0.2 + rnd.uniform(-0.03, 0.03)
            first = first if first is not None else t
            if c.add(True, now=t):
                assert 0.8 <= t - first <= 1.05
                break
        else:
            raise AssertionError("never fired")


# -------------------------------------------------------------------- camera

class _FakeCap:
    def __init__(self, *a):
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, *a):
        return True

    def read(self):
        time.sleep(0.005)
        return True, NOISE

    def release(self):
        self.opened = False


def test_camera_thread_delivers_fresh_frames():
    real = cv2.VideoCapture
    cv2.VideoCapture = _FakeCap
    try:
        cam = pg.Camera(0, 640)
        assert cam.start()
        _, s1 = cam.latest()
        time.sleep(0.15)
        _, s2 = cam.latest()
        assert s2 > s1 + 5
        cam.stop()
        assert not cam.alive()
        assert threading.active_count() == 1
    finally:
        cv2.VideoCapture = real


# ----------------------------------------------------------------- main loop

class _ScriptedCam:
    """Stands in for pg.Camera; hands out frames from a shared list."""
    def __init__(self, script):
        self.script = script
        self.seq = 0
        self.fps = 30.0
        self.stopped = False

    def latest(self):
        if not self.script:
            raise KeyboardInterrupt        # how the real loop is stopped
        self.seq += 1
        return self.script.pop(0), self.seq

    def alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


def _run(script, extra, calls):
    frames = list(script)

    def fake_open(index, width, attempts=3, wait=2.0):
        calls["open_camera"] += 1
        return _ScriptedCam(frames)

    pg.open_camera = fake_open
    pg.open_photo_booth = lambda: calls.__setitem__("pb_open", calls["pb_open"] + 1) or True
    pg.close_photo_booth = lambda: calls.__setitem__("pb_close", calls["pb_close"] + 1) or True
    pg.open_file = lambda p: calls.__setitem__("open_file", calls["open_file"] + 1) or True
    pg.beep = lambda: calls.__setitem__("beep", calls["beep"] + 1)
    pg.LOG_FILE = TMP / "screenpeek.log"
    pg.watch(pg.parse_args(["--interval", "0", "--confirm", "0.05", "--handover", "0",
                            "--show-for", "0"] + extra))


def _calls():
    return dict(open_camera=0, pb_open=0, pb_close=0, open_file=0, beep=0)


def test_full_trigger_sequence():
    d, c = out_dir("a"), _calls()
    _run([ONE] * 3 + [TWO] * 12, ["--dir", str(d), "--sound", "--cooldown", "300"], c)
    shots = list(d.glob("*.jpg"))
    assert len(shots) == 1 and "2faces" in shots[0].name
    assert c["pb_open"] == 1 and c["open_file"] == 1 and c["beep"] == 1
    assert c["open_camera"] == 2, "camera released for Photo Booth, then reopened"
    rows = list(csv.reader((d / "peeks.csv").open()))
    assert rows[0] == ["timestamp", "faces", "snapshot"] and rows[1][1] == "2"


def test_rearms_when_peeker_leaves():
    d, c = out_dir("b"), _calls()
    _run([ONE] * 3 + [TWO] * 12 + [TWO] * 12 + [ONE] * 25 + [TWO] * 14,
         ["--dir", str(d), "--rearm", "0.1", "--cooldown", "100"], c)
    assert len(list(d.glob("*.jpg"))) == 2
    log = pg.LOG_FILE.read_text()
    assert "armed again (extra face left)" in log and "trigger #2" in log


def test_rearms_on_cooldown_if_peeker_stays():
    d, c = out_dir("c"), _calls()
    _run([TWO] * 80, ["--dir", str(d), "--rearm", "100", "--cooldown", "0.25"], c)
    assert len(list(d.glob("*.jpg"))) >= 2
    assert "armed again (cooldown over)" in pg.LOG_FILE.read_text()


def test_min_faces_three_ignores_two_people():
    d, c = out_dir("d"), _calls()
    _run([TWO] * 12, ["--dir", str(d), "--min-faces", "3"], c)
    assert not list(d.glob("*.jpg")) and c["pb_open"] == 0


def test_silent_mode_launches_nothing():
    d, c = out_dir("e"), _calls()
    _run([TWO] * 12, ["--dir", str(d), "--no-photo-booth", "--no-open-snapshot"], c)
    assert len(list(d.glob("*.jpg"))) == 1
    assert c["pb_open"] == 0 and c["open_file"] == 0 and c["open_camera"] == 1


def test_keeps_retrying_when_camera_is_busy():
    d, c = out_dir("f"), _calls()
    frames = [TWO] * 12 + [ONE] * 25 + [TWO] * 14
    real_sleep = time.sleep
    pg.time.sleep = lambda s: real_sleep(min(s, 0.01))

    def flaky_open(index, width, attempts=3, wait=2.0):
        c["open_camera"] += 1
        return None if 2 <= c["open_camera"] <= 6 else _ScriptedCam(frames)

    try:
        pg.open_camera = flaky_open
        pg.open_photo_booth = lambda: True
        pg.close_photo_booth = lambda: True
        pg.open_file = lambda p: True
        pg.LOG_FILE = TMP / "screenpeek.log"
        pg.watch(pg.parse_args(["--interval", "0", "--confirm", "0.05", "--handover", "0",
                                "--show-for", "0", "--dir", str(d),
                                "--rearm", "0.1", "--cooldown", "100"]))
    finally:
        pg.time.sleep = real_sleep
    assert c["open_camera"] >= 7
    assert len(list(d.glob("*.jpg"))) == 2, "must keep working once the camera is back"
    assert "camera is back" in pg.LOG_FILE.read_text()


def test_snapshot_never_overwrites():
    d = out_dir("g")
    faces = [(10, 10, 50, 50, 0.9)] * 2
    a = pg.save_snapshot(TWO, faces, d)
    b = pg.save_snapshot(TWO, faces, d)
    assert a != b and a.exists() and b.exists()


def test_self_check_reports_camera_permission_problem():
    pg.open_camera = lambda *a, **k: None
    try:
        pg.self_check(pg.parse_args(["--check", "--dir", str(out_dir("h"))]))
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert "Privacy & Security" in str(exc)


# --------------------------------------------------------------------- runner

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:               # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
