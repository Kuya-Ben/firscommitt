"""
utils.py
========
Shared helpers for the Smart Face Attendance Monitoring System.

Everything that more than one script needs lives here:
    1. Project paths and tunable settings
    2. Image operations (resize, BGR->RGB, grayscale, blur)
    3. Haar Cascade face detection helpers
    4. Motion detection (bonus topic)
    5. Attendance CSV handling (no duplicate entries per day)
    6. Drawing helpers (boxes, labels, date/time overlay)

Keeping the helpers in one file keeps main.py short and easy to read.
"""

import csv
import importlib
import os
import re
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# 1. PATHS AND SETTINGS
# =============================================================================

# Folder that contains this file = the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
ENCODINGS_PATH = os.path.join(MODELS_DIR, "encodings.pickle")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "Attendance.csv")

# Column names of Attendance.csv
CSV_HEADER = ["Name", "Date", "Time", "Status"]

# ---- Tunable settings (change these to tweak speed / accuracy) --------------
FRAME_SCALE = 0.5          # Detect faces on a frame that is 50% of the size (faster)
PROCESS_EVERY_N = 2        # Run the heavy pipeline on every 2nd frame only
MATCH_TOLERANCE = 0.5      # Max face distance to accept a match (lower = stricter)
CONFIRM_FRAMES = 3         # Same person must be recognised this many times in a row
                           # before attendance is written (avoids false positives)

# Haar Cascade parameters
HAAR_SCALE_FACTOR = 1.1    # How much the image is shrunk at each detection pass
HAAR_MIN_NEIGHBORS = 5     # Higher = fewer false detections, but may miss faces
HAAR_MIN_SIZE = (30, 30)   # Ignore faces smaller than this (in the SMALL frame)

# Motion detection parameters
MOTION_MIN_PERCENT = 0.8   # % of pixels that must change to count as "motion"
IDLE_TIMEOUT = 3.0         # Keep detecting for this many seconds after last face seen
HEARTBEAT_EVERY = 10       # Even when idle, check for faces every 10th processed frame

# Colours are in BGR order (OpenCV's default), not RGB!
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_HEIGHT = 70         # Height (pixels) of the information panel at the top


# =============================================================================
# 2. PROJECT SETUP HELPERS
# =============================================================================

def ensure_folders():
    """Create all project folders if they do not exist yet."""
    for folder in (DATASET_DIR, MODELS_DIR, ATTENDANCE_DIR, SCREENSHOT_DIR):
        os.makedirs(folder, exist_ok=True)


def ensure_cascade_file():
    """
    Make sure models/haarcascade_frontalface_default.xml exists.
    If it is missing, copy the copy that ships with the opencv-python package.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(CASCADE_PATH):
        bundled = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(bundled):
            shutil.copy(bundled, CASCADE_PATH)
            print(f"[INFO] Copied Haar cascade to {CASCADE_PATH}")
        else:
            raise FileNotFoundError(
                "haarcascade_frontalface_default.xml not found. Download it from the "
                "OpenCV GitHub repository (data/haarcascades) and place it in models/."
            )


def load_cascade():
    """Load the Haar Cascade classifier used for face detection."""
    ensure_cascade_file()
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")
    return cascade


_face_recognition_lib = None


def get_face_recognition_lib():
    """
    Import the REAL 'face_recognition' library (dlib based).

    WHY THIS FUNCTION EXISTS
    ------------------------
    The assignment requires a project file called face_recognition.py. Python
    always looks in the script's own folder first, so a plain
    `import face_recognition` would import OUR file instead of the library
    (a "name shadowing" problem). To avoid this we temporarily hide the project
    folder from sys.path, import the real library, then put everything back.
    """
    global _face_recognition_lib
    if _face_recognition_lib is not None:
        return _face_recognition_lib

    saved_path = list(sys.path)
    saved_module = sys.modules.pop("face_recognition", None)   # our own file, if loaded
    project = os.path.abspath(BASE_DIR)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != project]
    try:
        library = importlib.import_module("face_recognition")
    except ImportError as error:
        raise ImportError(
            "The 'face_recognition' library is not installed. Run:\n"
            "    pip install -r requirements.txt\n"
            "(See README.md > Troubleshooting if dlib fails to install.)\n"
            f"Original error: {error}"
        )
    finally:
        # Restore the original import state.
        sys.path = saved_path
        sys.modules.pop("face_recognition", None)
        if saved_module is not None:
            sys.modules["face_recognition"] = saved_module

    _face_recognition_lib = library
    return library


def open_camera(index=0, width=640, height=480):
    """Open the webcam and request a 640x480 resolution."""
    # CAP_DSHOW opens the camera much faster on Windows.
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (index {index}). Check that it is connected, "
            "not used by another app, and that camera permission is allowed. "
            "Try --camera 1 for an external webcam."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# =============================================================================
# 3. NAME HELPERS
# =============================================================================

def sanitize_name(raw_name):
    """'  John  Doe! ' -> 'John_Doe' (safe to use as a folder name)."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", raw_name).strip()
    return re.sub(r"\s+", "_", cleaned)


def folder_to_display_name(folder_name):
    """'John_Doe' -> 'John Doe' (nice name shown on screen and in the CSV)."""
    return folder_name.replace("_", " ")


# =============================================================================
# 4. IMAGE OPERATIONS  (resize / BGR->RGB / grayscale / blur)
# =============================================================================

def resize_frame(frame, scale):
    """Resize a frame by a scale factor (0.5 = half size). Smaller = faster."""
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame):
    """
    OpenCV stores colours as Blue-Green-Red, but the face_recognition library
    (and most other libraries) expect Red-Green-Blue. We must convert.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def to_gray(frame):
    """Convert a BGR colour image to grayscale (Haar Cascades work on gray)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def enhance_gray(gray):
    """
    Prepare a grayscale image for Haar detection:
      * Gaussian blur removes small sensor noise (fewer false detections)
      * Histogram equalisation spreads brightness so dim rooms work better
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.equalizeHist(blurred)


# =============================================================================
# 5. HAAR CASCADE FACE DETECTION
# =============================================================================

def detect_faces_haar(cascade, gray):
    """
    Detect faces with the Haar Cascade classifier.
    Returns a list of (x, y, w, h) boxes in the coordinates of `gray`.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=HAAR_MIN_SIZE,
    )
    # detectMultiScale returns an empty tuple when nothing is found.
    return [tuple(int(v) for v in f) for f in faces]


def boxes_to_locations(boxes, scale):
    """
    Convert Haar boxes (x, y, w, h) found on the SMALL frame into the
    (top, right, bottom, left) format used by face_recognition, scaled back up
    to the FULL-size frame (so recognition uses the sharper full-size image).
    """
    locations = []
    for (x, y, w, h) in boxes:
        top = int(y / scale)
        right = int((x + w) / scale)
        bottom = int((y + h) / scale)
        left = int(x / scale)
        locations.append((top, right, bottom, left))
    return locations


def sharpness(gray_face):
    """
    Variance of the Laplacian = a classic blur measure.
    High value = sharp image, low value = blurry image.
    """
    gray_face = cv2.resize(gray_face, (100, 100))      # normalise size first
    return cv2.Laplacian(gray_face, cv2.CV_64F).var()


# =============================================================================
# 6. MOTION DETECTION (bonus topic)
# =============================================================================

class MotionDetector:
    """
    Simple frame-differencing motion detector.

    Idea: if the current frame is very different from the previous frame,
    something moved. We use it to "wake up" the face pipeline only when
    somebody is there, which saves CPU when the room is empty.
    """

    def __init__(self, min_percent=MOTION_MIN_PERCENT):
        self.previous = None
        self.min_percent = min_percent

    def update(self, gray):
        """Feed a grayscale frame. Returns (is_moving, percent_of_changed_pixels)."""
        # Blur first so tiny camera noise is not counted as movement.
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.previous is None:
            self.previous = blurred
            return False, 0.0

        diff = cv2.absdiff(self.previous, blurred)              # per-pixel difference
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # keep big changes only
        mask = cv2.dilate(mask, None, iterations=2)             # fill small holes
        percent = 100.0 * cv2.countNonZero(mask) / mask.size
        self.previous = blurred
        return percent > self.min_percent, percent


class FPSCounter:
    """Smoothed frames-per-second counter."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.previous = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        delta = now - self.previous
        self.previous = now
        if delta > 0:
            instant = 1.0 / delta
            # Exponential moving average -> number does not jump around.
            self.fps = instant if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * instant
        return self.fps


# =============================================================================
# 7. ATTENDANCE (CSV) HANDLING
# =============================================================================

def init_attendance_file():
    """Create Attendance.csv with a header row if it does not exist."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def load_attendance_df():
    """Read the whole attendance file into a pandas DataFrame."""
    init_attendance_file()
    try:
        return pd.read_csv(ATTENDANCE_FILE, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CSV_HEADER)


def get_today_records(date_str=None):
    """
    Return {name: time} for everybody already marked present on `date_str`
    (default = today). Used to prevent duplicate entries.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    df = load_attendance_df()
    today = df[df["Date"] == date_str]
    return dict(zip(today["Name"], today["Time"]))


def mark_attendance(name, today_records):
    """
    Write one attendance row (Name, Date, Time, Status) unless the person is
    already present today.

    Returns (newly_marked, time_string).
        newly_marked = True  -> a new row was written just now
        newly_marked = False -> already marked earlier (or the file was locked)
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Rule: each person is marked only once per day.
    if name in today_records:
        return False, today_records[name]

    # Double-check the file itself (another program may have written to it).
    records_on_disk = get_today_records(date_str)
    if name in records_on_disk:
        today_records[name] = records_on_disk[name]
        return False, records_on_disk[name]

    try:
        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, date_str, time_str, "Present"])
    except PermissionError:
        # On Windows this happens when Attendance.csv is open in Excel.
        print("[WARNING] Cannot write Attendance.csv - close it in Excel. Will retry.")
        return False, None

    today_records[name] = time_str
    print(f"[ATTENDANCE] {name} marked Present on {date_str} at {time_str}")
    return True, time_str


# =============================================================================
# 8. DRAWING HELPERS (image analysis & transformation: boxes, labels, overlay)
# =============================================================================

def put_text_bg(img, text, org, scale=0.55, color=WHITE, bg=BLACK, thickness=1, pad=4):
    """Draw text on top of a filled rectangle so it is readable on any background."""
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - text_h - pad), (x + text_w + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return text_w


def draw_face_box(frame, location, name, confidence, status, color):
    """
    Draw the bounding box, the name label (with confidence) and the
    attendance status text for one face.
    `location` is (top, right, bottom, left) in full-frame pixels.
    """
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Name label above the box (inside the box if there is no room above).
    label = f"{name} ({confidence * 100:.0f}%)" if name != "Unknown" else "Unknown"
    label_y = top - 10 if top - 10 > HEADER_HEIGHT + 15 else top + 22
    put_text_bg(frame, label, (left, label_y), scale=0.6, color=WHITE, bg=color)

    # Attendance status text below the box.
    status_y = min(bottom + 24, frame.shape[0] - 40)
    put_text_bg(frame, status, (left, status_y), scale=0.5, color=WHITE, bg=BLACK)


def draw_header(frame, now, registered, present, fps, motion_text):
    """
    Draw the translucent information panel at the top of the video:
    title, date, time, registered count, present count, FPS and motion status.
    """
    height, width = frame.shape[:2]

    # Semi-transparent black bar: copy the frame, draw on the copy, then blend.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, HEADER_HEIGHT), BLACK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Left side: title + date and time
    cv2.putText(frame, "Smart Face Attendance", (10, 24), FONT, 0.65, YELLOW, 2, cv2.LINE_AA)
    date_time = now.strftime("%Y-%m-%d   %H:%M:%S")
    cv2.putText(frame, date_time, (10, 54), FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    # Right side: counters
    right_x = max(width - 250, 330)
    cv2.putText(frame, f"Registered: {registered}", (right_x, 20), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Present today: {present}", (right_x, 40), FONT, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f} | Motion: {motion_text}", (right_x, 60), FONT, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_footer(frame, toast_text=""):
    """Bottom bar with keyboard help and an optional message (toast)."""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 26), (width, height), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "q: quit | s: screenshot | m: motion on/off", (10, height - 8),
                FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    if toast_text:
        put_text_bg(frame, toast_text, (10, height - 40), scale=0.55, color=BLACK, bg=GREEN)
