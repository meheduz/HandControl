"""Control your Mac with hand gestures using the webcam (MediaPipe Tasks API).

Gestures
  Index up                    move cursor
  Thumb + index pinch         click (quick), double click (two quick),
                              drag (hold the pinch and move)
  Thumb + middle pinch        right click
  Index + middle up           scroll (move hand up/down)
  Index + middle + ring up    volume (move hand up/down)
  Open palm, swipe fast       left/right: switch desktop, up: Mission Control,
                              down: App Expose
  Fist held 1 second          lock / unlock (ignores everything while locked)
  Keys in the window          q quit, l lock/unlock, c recenter control box

Setup
  pip install opencv-python mediapipe pyautogui
Allow Camera and Accessibility for your Terminal/IDE in macOS settings.
Swipe uses the ctrl+arrow shortcuts, enabled by default in Keyboard Shortcuts.
"""
import collections
import math
import os
import ssl
import subprocess
import threading
import time
import urllib.request

import certifi
import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from Quartz import (CGEventCreate, CGEventCreateMouseEvent,
                    CGEventCreateScrollWheelEvent, CGEventGetLocation,
                    CGEventPost, CGEventSetIntegerValueField,
                    kCGEventLeftMouseDown, kCGEventLeftMouseDragged,
                    kCGEventLeftMouseUp, kCGEventMouseMoved,
                    kCGEventRightMouseDown, kCGEventRightMouseUp,
                    kCGHIDEventTap, kCGMouseButtonLeft, kCGMouseButtonRight,
                    kCGMouseEventClickState, kCGScrollEventUnitPixel)

MODEL = "hand_landmarker.task"
URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL):
    print("Downloading hand model...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=ctx) as r, open(MODEL, "wb") as f:
        f.write(r.read())

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SW, SH = pyautogui.size()

# ---- tuning ----
MARGIN = 0.2          # larger = smaller hand area covers the whole screen
EASE_MIN = 0.12       # glide when close to the target (smooth, precise)
EASE_MAX = 0.6        # glide when far from the target (fast)
EASE_FAR = 500        # px distance at which the glide hits EASE_MAX
AUTO_LOCK = 20        # seconds without a hand before auto-lock (0 = off)
PINCH_CONFIRM = 2     # frames a pinch must persist before it clicks
MOVER_HZ = 120
MIN_CUTOFF = 1.5      # filter: lower = steadier when still
BETA = 0.01           # filter: higher = less lag when moving fast
DEADZONE = 2          # px
PINCH_ON = 0.20
PINCH_OFF = 0.35
PINCH_REACH = 1.1     # finger must be extended this much to count as a pinch
UP_RATIO = 1.25
HOLD = 2              # frames a gesture must persist before it switches
DRAG_DELAY = 0.25     # seconds of holding a pinch before it becomes a drag
DBL_T = 0.45          # max gap for a double click
DBL_DIST = 30         # px
LOCK_HOLD = 1.0       # seconds of fist to toggle lock
LOST_GRACE = 0.3      # seconds before a lost hand releases the mouse
SCROLL_K = 1500
SCROLL_DEAD = 0.0015
VOL_K = 150
SWIPE_ACTION = "desktop"  # "desktop" = ctrl+arrows, "app" = cmd+tab / cmd+shift+tab
SWIPE_WIN = 0.5       # seconds
SWIPE_D = 0.15        # fraction of the frame
SWIPE_COOLDOWN = 0.8
SWIPE_INVERT = False  # True flips left/right
DETECT_W, DETECT_H = 320, 240
SHOW = True


# ---- macOS input ----
def cursor_pos():
    p = CGEventGetLocation(CGEventCreate(None))
    return p.x, p.y


def post(kind, x, y, button=kCGMouseButtonLeft, n=1):
    ev = CGEventCreateMouseEvent(None, kind, (x, y), button)
    CGEventSetIntegerValueField(ev, kCGMouseEventClickState, n)
    CGEventPost(kCGHIDEventTap, ev)


def scroll(dy):
    CGEventPost(kCGHIDEventTap,
                CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 1, dy))


def get_volume():
    try:
        out = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=2).stdout.strip()
        return float(out)
    except Exception:
        return 50.0


KEYCODE = {"left": 123, "right": 124, "down": 125, "up": 126}


def swipe_key(key):
    """Send the swipe shortcut through System Events (more reliable than pyautogui)."""
    if SWIPE_ACTION == "app" and key in ("left", "right"):
        mods = "command down" if key == "right" else "{command down, shift down}"
        script = f"tell application \"System Events\" to key code 48 using {mods}"
    else:
        script = (f"tell application \"System Events\" to key code "
                  f"{KEYCODE[key]} using control down")
    subprocess.Popen(["osascript", "-e", script],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_volume(v):
    subprocess.Popen(["osascript", "-e", f"set volume output volume {int(v)}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- helpers ----
class OneEuro:
    def __init__(self):
        self.x = self.dx = self.t = None

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx += self.alpha(1.0, dt) * (dx - self.dx)
        a = self.alpha(MIN_CUTOFF + BETA * abs(self.dx), dt)
        self.x += a * (x - self.x)
        self.t = t
        return self.x


class Cam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.run = True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.run:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

    def stop(self):
        self.run = False
        self.cap.release()


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(lm, tip, pip):
    return dist(lm[0], lm[tip]) > dist(lm[0], lm[pip]) * UP_RATIO


# shared state between the detection loop and the cursor thread
cx0, cy0 = cursor_pos()
S = {"moving": False, "drag": False, "tx": cx0, "ty": cy0,
     "cx": cx0, "cy": cy0, "run": True}


def mover():
    """Glides the cursor toward the target at a steady rate."""
    while S["run"]:
        if S["moving"]:
            dx, dy = S["tx"] - S["cx"], S["ty"] - S["cy"]
            if math.hypot(dx, dy) > DEADZONE:
                e = min(EASE_MAX, max(EASE_MIN, math.hypot(dx, dy) / EASE_FAR))
                S["cx"] += dx * e
                S["cy"] += dy * e
                kind = kCGEventLeftMouseDragged if S["drag"] else kCGEventMouseMoved
                post(kind, S["cx"], S["cy"])
        time.sleep(1 / MOVER_HZ)


opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(opts)

CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

cam = Cam()
threading.Thread(target=mover, daemon=True).start()
fx, fy = OneEuro(), OneEuro()

pinching = rpinch = False
pinch_t, pinch_n = 0.0, 0
ccx, ccy = 0.5, 0.5   # center of the control box, follows your hand
last_up_t, last_up_pos, last_n = -10.0, (0, 0), 1
locked = False
fist_t, fist_done = None, False
mode, cand, cand_n = "none", "none", 0
prev_y = None
hist = collections.deque()
swipe_cd = 0.0
vol = get_volume()
vol_sent, vol_t = int(vol), 0.0
notice, notice_t = "", -10.0
last_seen = 0.0
t0 = time.time()
last_ts = -1
fps, fps_t, fps_n = 0, time.time(), 0

while True:
    if cam.frame is None:
        time.sleep(0.01)
        continue
    frame = cv2.flip(cam.frame, 1)
    small = cv2.resize(frame, (DETECT_W, DETECT_H))
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    now = time.time() - t0
    ts = max(int(now * 1000), last_ts + 1)
    last_ts = ts
    res = detector.detect_for_video(img, ts)
    lm = res.hand_landmarks[0] if res.hand_landmarks else None
    raw = "none"

    if lm:
        last_seen = now
        size = max(dist(lm[0], lm[9]), 1e-6)
        r_i = dist(lm[4], lm[8]) / size
        r_m = dist(lm[4], lm[12]) / size
        idx = finger_up(lm, 8, 6)
        mid = finger_up(lm, 12, 10)
        ring = finger_up(lm, 16, 14)
        pinky = finger_up(lm, 20, 18)
        reach_i = dist(lm[0], lm[8]) > dist(lm[0], lm[5]) * PINCH_REACH
        reach_m = dist(lm[0], lm[12]) > dist(lm[0], lm[9]) * PINCH_REACH

        if not locked:
            # left pinch: mouse down right away, so hold = drag, quick = click
            pinch_n = pinch_n + 1 if (r_i < PINCH_ON and reach_i) else 0
            if not pinching and not rpinch and pinch_n >= PINCH_CONFIRM:
                pinching, pinch_t = True, now
                x, y = S["cx"], S["cy"]
                dbl = (now - last_up_t < DBL_T and last_n == 1 and
                       math.hypot(x - last_up_pos[0], y - last_up_pos[1]) < DBL_DIST)
                last_n = 2 if dbl else 1
                post(kCGEventLeftMouseDown, x, y, kCGMouseButtonLeft, last_n)
            elif pinching and r_i > PINCH_OFF:
                pinching = False
                S["drag"] = False
                post(kCGEventLeftMouseUp, S["cx"], S["cy"], kCGMouseButtonLeft, last_n)
                last_up_t, last_up_pos = now, (S["cx"], S["cy"])

            # right click: thumb + middle pinch
            if not pinching:
                if not rpinch and r_m < PINCH_ON and reach_m and r_i > PINCH_OFF:
                    rpinch = True
                    post(kCGEventRightMouseDown, S["cx"], S["cy"], kCGMouseButtonRight)
                    post(kCGEventRightMouseUp, S["cx"], S["cy"], kCGMouseButtonRight)
                elif rpinch and r_m > PINCH_OFF:
                    rpinch = False

        if pinching:
            raw = "pinch"
        elif rpinch:
            raw = "rclick"
        elif not (idx or mid or ring or pinky):
            raw = "fist"
        elif idx and not (mid or ring or pinky):
            raw = "move"
        elif idx and mid and not (ring or pinky):
            raw = "scroll"
        elif idx and mid and ring and not pinky:
            raw = "volume"
        elif idx and mid and ring and pinky:
            raw = "palm"

        # fist held = lock toggle
        if raw == "fist":
            if fist_t is None:
                fist_t = now
            elif not fist_done and now - fist_t >= LOCK_HOLD:
                fist_done = True
                locked = not locked
                notice, notice_t = ("LOCKED" if locked else "UNLOCKED"), now
        else:
            fist_t, fist_done = None, False
    else:
        S["moving"] = False
        fx.reset()
        fy.reset()
        prev_y = None
        hist.clear()
        fist_t, fist_done = None, False
        if now - last_seen > LOST_GRACE:
            if pinching:
                pinching = False
                S["drag"] = False
                post(kCGEventLeftMouseUp, S["cx"], S["cy"], kCGMouseButtonLeft, last_n)
            rpinch = False
            mode = cand = "none"
            cand_n = 0
            if AUTO_LOCK and not locked and now - last_seen > AUTO_LOCK:
                locked = True
                notice, notice_t = "LOCKED (idle)", now

    # debounce so a gesture must hold for HOLD frames before it switches
    if lm:
        if raw == cand:
            cand_n += 1
        else:
            cand, cand_n = raw, 1
        if cand_n >= HOLD and cand != mode:
            mode = cand
            prev_y = None
            hist.clear()

    if lm and not locked:
        # cursor tracking: index knuckle joint barely moves during a pinch
        if (pinching or mode == "move") and not rpinch:
            # the control box gets pushed by your hand, so you never reach an edge
            W = 1 - 2 * MARGIN
            px, py = lm[6].x, lm[6].y
            ccx += max(0.0, px - (ccx + W / 2)) - max(0.0, (ccx - W / 2) - px)
            ccy += max(0.0, py - (ccy + W / 2)) - max(0.0, (ccy - W / 2) - py)
            ccx = min(max(ccx, W / 2), 1 - W / 2)
            ccy = min(max(ccy, W / 2), 1 - W / 2)
            x = (px - (ccx - W / 2)) / W
            y = (py - (ccy - W / 2)) / W
            S["tx"] = fx(min(max(x, 0), 1) * SW, now)
            S["ty"] = fy(min(max(y, 0), 1) * SH, now)
            if pinching:
                S["drag"] = now - pinch_t > DRAG_DELAY
                S["moving"] = S["drag"]
            else:
                S["moving"] = raw == "move"
        else:
            S["moving"] = False
            if not rpinch:
                fx.reset()
                fy.reset()

        if mode == "scroll" and raw == "scroll":
            y = (lm[8].y + lm[12].y) / 2
            if prev_y is not None and abs(prev_y - y) > SCROLL_DEAD:
                scroll(int((prev_y - y) * SCROLL_K))
            prev_y = y
        elif mode == "volume" and raw == "volume":
            y = lm[8].y
            if prev_y is not None:
                vol = min(100.0, max(0.0, vol + (prev_y - y) * VOL_K))
                if int(vol) != vol_sent and now - vol_t > 0.12:
                    vol_sent, vol_t = int(vol), now
                    set_volume(vol_sent)
            prev_y = y
            notice, notice_t = f"volume {vol_sent}%", now
        else:
            prev_y = None

        if mode == "palm" and raw == "palm":
            hist.append((now, lm[9].x, lm[9].y))
            while hist and now - hist[0][0] > SWIPE_WIN:
                hist.popleft()
            if now > swipe_cd and len(hist) >= 3:
                dx = hist[-1][1] - hist[0][1]
                dy = hist[-1][2] - hist[0][2]
                key = None
                if abs(dx) > SWIPE_D and abs(dx) > 2 * abs(dy):
                    key = "right" if (dx > 0) != SWIPE_INVERT else "left"
                elif abs(dy) > SWIPE_D and abs(dy) > 2 * abs(dx):
                    key = "up" if dy < 0 else "down"
                if key:
                    swipe_key(key)
                    swipe_cd = now + SWIPE_COOLDOWN
                    hist.clear()
                    notice, notice_t = f"swipe {key}", now
        else:
            hist.clear()
    elif locked:
        S["moving"] = False
        prev_y = None
        hist.clear()

    # keep our idea of the cursor in sync if you also use the real mouse
    if not S["moving"] and not pinching:
        S["cx"], S["cy"] = cursor_pos()

    fps_n += 1
    if time.time() - fps_t >= 1:
        fps, fps_n, fps_t = fps_n, 0, time.time()

    if SHOW:
        h, w = frame.shape[:2]
        half = (1 - 2 * MARGIN) / 2
        cv2.rectangle(frame, (int((ccx - half) * w), int((ccy - half) * h)),
                      (int((ccx + half) * w), int((ccy + half) * h)),
                      (90, 90, 90), 1)
        if lm:
            for a, b in CONN:
                cv2.line(frame, (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)), (0, 255, 0), 2)
            cv2.circle(frame, (int(lm[6].x * w), int(lm[6].y * h)), 8,
                       (0, 200, 255), -1)
        if locked:
            label, color = "LOCKED", (0, 0, 255)
        elif S["drag"]:
            label, color = "drag", (0, 200, 255)
        else:
            label, color = mode, (0, 255, 0)
        cv2.putText(frame, f"{label}  {fps} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if time.time() - t0 - notice_t < 1.2:
            cv2.putText(frame, notice, (10, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        cv2.putText(frame, "fist 1s: lock   q: quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Hand Control", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        ccx, ccy = 0.5, 0.5
    if k == ord("l"):
        locked = not locked
        notice, notice_t = ("LOCKED" if locked else "UNLOCKED"), now

if pinching:
    post(kCGEventLeftMouseUp, S["cx"], S["cy"], kCGMouseButtonLeft, last_n)
S["run"] = False
cam.stop()
cv2.destroyAllWindows()
